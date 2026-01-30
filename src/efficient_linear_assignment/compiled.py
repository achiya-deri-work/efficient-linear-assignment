
import torch
import torch._dynamo

# Recommended Options for Performance
options = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "triton.cudagraphs": False,
    "max_autotune_gemm": True,
    "coordinate_descent_tuning": True,
    "aggressive_fusion": True,
    "max_autotune_pointwise": True,
}

class CompiledKernelDispatcher:
    """
    Manages specialized compiled kernels for different input shapes.
    Prevents 'recompile_limit' issues by creating fresh function closures for each shape.
    """
    def __init__(self, kernel_factory, options=None):
        self.kernel_factory = kernel_factory
        self.options = options or globals().get('options') # Use module level options if not provided
        self.cache = {}
    
    def __call__(self, *args, **kwargs):
        # Infer shape from first argument (assumed Tensor)
        x = args[0]
        if x.ndim == 2:
            shape = (1, x.shape[0], x.shape[1])
        else:
            shape = x.shape
            
        # Key: (B, N, M)
        # We might need looser key if N,M vary but B doesn't matter?
        # Typically B affects Triton configs. Be specific.
        key = tuple(shape)
        
        if key not in self.cache:
            # Create fresh compiled closure
            self.cache[key] = self.kernel_factory(key, self.options)
            
        return self.cache[key](*args, **kwargs)

# --- Sinkhorn Factory ---
def _create_sinkhorn(shape, options):
    @torch.compiler.nested_compile_region
    def inner_step(f, g, M_eps, log_mu, log_nu):
        f = log_mu.unsqueeze(-1) - torch.logsumexp(M_eps + g, dim=2, keepdim=True)
        g = log_nu.unsqueeze(1) - torch.logsumexp(M_eps + f, dim=1, keepdim=True)
        return f, g

    def outer(C, mu=None, nu=None, epsilon=0.1, num_iters=20, **kwargs):
        if C.ndim == 2: C = C.unsqueeze(0)
        B, N, M = C.shape
        device = C.device
        dtype = C.dtype
        
        if mu is None: mu = torch.ones(B, N, device=device, dtype=dtype) / N
        if nu is None: nu = torch.ones(B, M, device=device, dtype=dtype) / M
        
        log_mu = torch.log(mu + 1e-8)
        log_nu = torch.log(nu + 1e-8)
        
        f = torch.zeros(B, N, 1, device=device, dtype=dtype)
        g = torch.zeros(B, 1, M, device=device, dtype=dtype)
        
        M_eps = -C / epsilon
        
        for _ in range(num_iters):
            f, g = inner_step(f, g, M_eps, log_mu, log_nu)
            
        return torch.exp(M_eps + f + g)

    return torch.compile(outer, dynamic=False, fullgraph=True, options=options)

# --- Dual Ascent Factory ---
def _create_dual_ascent(shape, options):
    @torch.compiler.nested_compile_region
    def inner_step(alpha, beta, C, mu, nu, epsilon):
        # Gradient Ascent Step Size
        # Matches CUDA backend: step = epsilon * 0.5
        # This is more stable than Newton for L2 Dual which has piecewise linear derivative.
        step_size = epsilon * 0.5
        
        # Row
        T = alpha + beta - C
        # P = ReLU(T) / epsilon
        # sum_P = sum(ReLU(T)) / epsilon
        # grad_alpha = mu - sum_P
        # alpha += step * grad_alpha
        
        # Use ReLU directly
        P = torch.relu(T) / epsilon
        current_sum = P.sum(dim=2, keepdim=True)
        grad_alpha = mu.unsqueeze(-1) - current_sum
        alpha = alpha + step_size * grad_alpha
        
        # Col
        T2 = alpha + beta - C
        P2 = torch.relu(T2) / epsilon
        current_sum2 = P2.sum(dim=1, keepdim=True)
        grad_beta = nu.unsqueeze(1) - current_sum2
        beta = beta + step_size * grad_beta
        
        return alpha, beta

    def outer(C, mu=None, nu=None, epsilon=1.0, num_iters=10, **kwargs):
        if C.ndim == 2: C = C.unsqueeze(0)
        B, N, M = C.shape
        device = C.device
        dtype = C.dtype
        
        if mu is None: mu = torch.ones(B, N, device=device, dtype=dtype) / N
        if nu is None: nu = torch.ones(B, M, device=device, dtype=dtype) / M
        
        alpha = torch.zeros(B, N, 1, device=device, dtype=dtype)
        beta = torch.zeros(B, 1, M, device=device, dtype=dtype)
        
        for _ in range(num_iters):
             alpha, beta = inner_step(alpha, beta, C, mu, nu, epsilon)
             
        return torch.relu(alpha + beta - C) / epsilon

    return torch.compile(outer, dynamic=False, fullgraph=True, options=options)

# --- Public Instances ---
# --- Auction Factory ---
def _create_auction(shape, options):
    # Static-Shape Dense Auction Step for Inductor
    # Avoids 'nonzero' to keep shapes static (B*N).
    @torch.compiler.nested_compile_region
    def inner_step(benefits, prices, assignment, obj_to_agent, epsilon):
        B, N, M = 1, benefits.shape[0], benefits.shape[1] # Flat benefits used inside
        
        # 1. Identify Unassigned
        # assignment: (N_total,)
        is_unassigned = (assignment == -1)
        
        # 2. Compute Values (All Agents)
        # vals = benefits - prices
        # prices: (M,) broadcast to (N, M)
        # Note: We must compute for all to keep shape static, but mask results
        vals = benefits - prices.unsqueeze(0)
        
        # 3. Top 2
        # (N, M) -> (N, 2)
        top2_vals, top2_idxs = torch.topk(vals, k=2, dim=1)
        
        best_val = top2_vals[:, 0]
        second_val = top2_vals[:, 1]
        best_obj_idx = top2_idxs[:, 0]
        
        # 4. Compute Bids
        # bid = price + val_best - val_second + eps
        # We only care about bids from UNASSIGNED agents.
        # But we compute for all.
        increments = best_val - second_val + epsilon
        target_prices = prices[best_obj_idx]
        bids = target_prices + increments
        
        # 5. Scatter Updates (Collision Resolution)
        # We need to simulate the "Election".
        # We only scatter bids from UNASSIGNED agents.
        # Mask: where(is_unassigned, bid, -inf)
        
        # Valid Bids:
        # We want to maximize price. Lower bids (or no bids) should not affect max.
        # Set invalid bids to -infinity
        valid_bids = torch.where(is_unassigned, bids, torch.tensor(-1e9, device=bids.device, dtype=bids.dtype))
        
        # Scatter Reduce 'amax' to prices
        # prices = max(prices, valid_bids scattered)
        # Warning: scattered tensor must be (M,).
        # We use scatter_reduce_ on a clone or output buffer?
        # Inductor supports scatter_reduce logic.
        
        # Indices to write to: best_obj_idx (N,)
        # We need to update 'prices'.
        # new_prices = prices.clone()
        # new_prices.scatter_reduce_(0, best_obj_idx, valid_bids, reduce='amax')
        # BUT this is complex to express purely functional without side-effects for nested region?
        # Actually nested_region allows functional tensor outputs.
        # scatter_reduce returns 'self'. 
        
        # Proposed Logic:
        # 1. Calculate max bid per object efficiently. Requires N->M reduce.
        #    This is hard to do without scatter.
        #    scatter_reduce_ is the way.
        
        # For simplicity in compiled graph, we return the bids and indices, 
        # and let the outer loop apply scatter? 
        # NO, outer loop is Python. Inner step must do heavy lifting.
        
        # Let's try functional scatter (tensor.scatter_reduce is in-place).
        # We return UPDATED prices and assignments.
    
        new_prices = prices.scatter_reduce(0, best_obj_idx, valid_bids, reduce='amax', include_self=True)
    
        # 6. Determine Winners
        # Who matched the new price?
        # bid == new_price[obj_idx]
        updated_obj_prices = new_prices[best_obj_idx]
        # Robust float check
        did_match = (torch.abs(bids - updated_obj_prices) < 1e-4) & is_unassigned
        
        # 7. Assignment update (Simulated Last-Writer-Wins via scatter)
        # We need to assign objects to agents.
        # Objects: best_obj_idx
        # Agents: arange(N)
        # Map: Object -> Agent.
        
        # Initialize winners with -1
        # obj_winners = full(M, -1)
        # obj_winners.scatter_(0, best_obj_idx[did_match], agent_idxs[did_match])
        
        # This part requires dynamic indexing (did_match filtering).
        # Masking is better:
        # We want to perform scatter for ALL, but masked?
        # scatter doesn't support mask arg directly.
        # We can use invalid indices (e.g. M) for scatter if we pad target? No.
        
        # OK, Compilation of Auction Collision Resolution is HARD.
        # For this task, we will simplify:
        # Return the bids and let outer (compiled fullgraph?) handle it?
        # No.
        
        # Fallback: We will just return the computed Bids and Indices, 
        # and let the Python loop handle the scatter/assignment logic.
        # The expensive part is TopK and memory fetch.
        return bids, best_obj_idx, is_unassigned

    def outer(C, epsilon=1.0, max_iter=500, **kwargs):
        if C.ndim == 2: C = C.unsqueeze(0)
        B, N, M = C.shape
        # Flatten
        flat_benefits = -C.view(-1, M)
        prices = torch.zeros(M, device=C.device, dtype=C.dtype) # Shared prices per batch? No, AuctionTorch handles batch offset?
        # Wait, usually indices are global (B*M).
        # Adapted simplified version:
        
        # REVERT: Writing a correct Compiled Auction from scratch in 5 mins is risky.
        # The user wants "compile for auction".
        # Best approach: Use torch.compile on the EXISTING AuctionTorch.solve but with dynamic=True
        # and maybe compile JUST the topk part.
        
        pass

# --- Revised Strategy ---
# We will NOT implement a full custom kernel here. 
# We will use torch.compile on a wrapper that calls AuctionTorch logic.
def _create_auction(shape, options):
    """
    Creates a Dense, Static-Shape Auction Solver compatible with Inductor.
    Avoids dynamic control flow (nonzero, unique) by using masked scatters.
    """
    @torch.compiler.nested_compile_region
    def inner_step(benefits, prices, assignment, obj_owner, epsilon):
        # Flattened shapes: benefits (N_total, M), prices (M,), assignment (N_total,), obj_owner (M,) (per batch offset handled externally or N_total=B*N)
        # Actually, for B>1, we need to handle batch offsets.
        # Simplest: Flatten everything to (B*N, M) world view.
        # But `prices` is (M,) if B=1? No, `prices` should be (B, M) -> (B*M).
        
        # benefits is (B, N, M)
        B, N, M = benefits.shape
        # Note: benefits is (B*N, M) strictly? NO.
        # If B>1, benefits (B, N, M). 
        # To compile efficently, we treat it as (B*N, M) sparse-like or just handle batch offsets manually?
        # The provided 'outer' flattens it.
        
        # 1. Identify Is_Active
        is_active = (assignment == -1)
        
        # 2. Compute Values (Broadcast prices)
        # prices: (B*M). We need to subtract prices from benefits.
        # Structure: Agents 0..N-1 talk to Objects 0..M-1. 
        # Agents N..2N-1 (Batch 1) talk to Objects M..2M-1 (Batch 1).
        # We need a map from Agent_Global_ID -> Batch_ID -> Object_Global_Offset.
        # This map is static. Passing it in?
        # Better: Assume pre-calculated 'prices_expanded' or use indexing.
        # Let's assume 'outer' prepares 'row_to_obj_offset' (N_total,).
        pass 
        # Actually, simple reshape allows broadcasting if (B, N, M).
        # But 'nested_compile_region' works best on flat tensors mostly? 
        # Let's try 3D input to function.
        
        # RE-DESIGN: Input (B, N, M). All tensors 3D?
        # prices (B, 1, M), assignment (B, N), obj_owner(B, 1, M).
        # This preserves batch isolation automatically.
        
        vals = benefits - prices.unsqueeze(1) # (B, N, M) - (B, 1, M) -> (B, N, M).
        
        # 3. Top 2
        top2_vals, top2_idxs = torch.topk(vals, k=2, dim=2)
        best_val = top2_vals[:, :, 0]    # (B, N)
        sec_val = top2_vals[:, :, 1]     # (B, N)
        best_obj_idx = top2_idxs[:, :, 0] # (B, N) (Local Index 0..M-1)
        
        # 4. Bids
        increments = best_val - sec_val + epsilon
        # Gather target prices: prices is (B, 1, M). best_obj_idx is (B, N).
        target_prices = prices.squeeze(1).gather(1, best_obj_idx) # (B, N)
        bids = target_prices + increments # (B, N)
        
        # 5. Scatter to Update Prices
        # We need to scatter 'bids' to 'prices' at 'best_obj_idx'.
        # Valid bids only (from active agents).
        valid_bids = torch.where(is_active, bids, torch.tensor(-1e9, device=bids.device, dtype=bids.dtype))
        
        # scatter_reduce(dim, index, src, reduce)
        # prices is (B, 1, M). We scatter on dim 2.
        # index needs to be (B, 1, N)? No, index must match src size (B, N).
        # We want to scatter (B, N) source to (B, M) dest, using (B, N) indices.
        # PyTorch scatter supports this for 3D if we unsqueeze prices to (B, N, M)? No.
        # Flattening to (B*N) and (B*M) is easier for scatter?
        # scatter dim=-1 works batch-wise if everything is (B, ...).
        # prices: (B, M). src: (B, N). index: (B, N). 
        # "scatter_reduce_: index and src must have same size". prices is dest.
        # prices.scatter_reduce_(1, best_obj_idx, valid_bids, reduce='amax')
        # This works!
        
        # Note: scatter_reduce_ is in-place. nesting might prefer functional?
        # prices = prices.scatter_reduce(...) for functional safety if needed.
        # But standard scatter_reduce_ returns self.
        
        # Update Prices
        # We need a clone if we want to check differences later? 
        old_prices = prices.clone() # Needed? Maybe just for convergence check.
        new_prices = prices.scatter_reduce(1, best_obj_idx, valid_bids, reduce='amax', include_self=True)
        
        # 6. Determine Winners (Tie-Breaking)
        # Who matched the new price?
        # Gather new prices back to agents
        updated_obj_prices = new_prices.gather(1, best_obj_idx)
        # Check match (float robust)
        is_winner = (torch.abs(bids - updated_obj_prices) < 1e-4) & is_active
        
        # Resolve multiple winners for same object
        # Scatter Agent IDs to Objects. Max ID wins.
        # Agent IDs: 0..N-1 repeated for each batch.
        # We need (B, N) tensor of IDs.
        agent_ids = torch.arange(benefits.shape[1], device=benefits.device).unsqueeze(0).expand(benefits.shape[0], -1)
        
        # Mask losers with -1
        valid_winner_ids = torch.where(is_winner, agent_ids, torch.tensor(-1, device=agent_ids.device, dtype=torch.long))
        
        # Scatter IDs to obj_owner (B, M)
        # obj_owner initialized to -1.
        # We use 'amax' because valid IDs >= 0, looser/inactive = -1.
        new_obj_owner = obj_owner.scatter_reduce(1, best_obj_idx, valid_winner_ids, reduce='amax', include_self=True)
        
        return new_prices, new_obj_owner

    def outer(C, epsilon=1.0, max_iter=500, **kwargs):
        # 1. Setup
        if C.ndim == 2: C = C.unsqueeze(0)
        B, N, M = C.shape
        device = C.device
        dtype = C.dtype
        
        # benefits = -C (Maximize net value)
        # Or standard Min-Cost Auction: Value = -C - price.
        # Here we follow: Value = Benefit - Price. Benefit = -C.
        # So we maximize (-C - Price).
        benefits = -C
        
        prices = torch.zeros(B, M, device=device, dtype=dtype)
        assignment = torch.full((B, N), -1, device=device, dtype=torch.long)
        obj_owner = torch.full((B, M), -1, device=device, dtype=torch.long)
        
        loop_iter = 0
        while loop_iter < max_iter:
            loop_iter += 1
            
            # 2. Compile Region Step
            # Returns updated (B, M) prices and (B, M) owners.
            new_prices, new_obj_owner = inner_step(benefits, prices, assignment, obj_owner, epsilon)
            
            # 3. Update Assignments (Dense Evicition Logic)
            # This part interacts with 'assignment' (B, N) using 'obj_owner' (B, M).
            # We can compile this too? Or keep it here.
            # To minimize graph breaks, keep heavy masking inside? 
            # But 'inner_step' inputs assignment for 'is_active'.
            # We need to update 'assignment' before next call.
            
            # Detect Changes
            # where owner changed AND wasn't -1
            changed_mask = (new_obj_owner != obj_owner) & (new_obj_owner != -1)
            
            # A. EVICT OLD OWNERS
            # Old owner of object j was obj_owner[j].
            # If changed_mask[j] is True, we must set assignment[obj_owner[j]] = -1.
            
            # Use 'No-Op Scatter' trick to avoid dynamic indexing.
            # safe_indices = clamp(obj_owner, 0, N-1)
            # update_vals = where(changed & (obj_owner!=-1), -1, assignment[safe_idx])
            # But wait: 'assignment[safe_idx]' reads current request. 
            # If we scatter -1, we rely on duplicate handling?
            # scatter_ dim 1 (Agents).
            # indices = obj_owner (B, M). src = -1.
            # We want to scatter into 'assignment' (B, N).
            
            # Indices: obj_owner (B, M).
            # Safe Indices: obj_owner.clamp(0)
            # Src: where(changed & valid_old, -1, assignment.gather(1, safe_ind))
            # Wait, scatter_ writes FROM src TO dest[index].
            # We want assignment[obj_owner[b,j]] = -1.
            
            # Construct update payload
            # We scan M objects. For each, we might update one agent.
            # Dest: assignment (B, N).
            # Indices: obj_owner (B, M).
            # Update: -1.
            
            # We need to preserve assignment for agents NOT being evicted.
            # scatter_ overwrites.
            # We just need to selectively write -1.
            # scatter(index, src)
            # But we can't 'skip' indices in dense scatter.
            # Trick: Write 'current_value' to 'index' if no change.
            #   assignment.scatter_(1, index, assignment.gather(1, index)) -> No-op.
            #   assignment.scatter_(1, index, -1) -> Evict.
            
            # So:
            # target_agents = obj_owner.clamp(min=0)
            # current_vals = assignment.gather(1, target_agents)
            # new_vals = torch.where(changed_mask & (obj_owner != -1), torch.tensor(-1, device=device), current_vals)
            # assignment.scatter_(1, target_agents, new_vals)
            
            target_agents = obj_owner.clamp(min=0)
            # Gather current assignment for these agents to execute no-ops
            current_vals = assignment.gather(1, target_agents)
            
            # Evict payload: -1 if changed and valid old owner. Else keep current.
            evict_vals = torch.where(changed_mask & (obj_owner != -1), torch.tensor(-1, device=device, dtype=torch.long), current_vals)
            
            assignment.scatter_(1, target_agents, evict_vals)
            
            # B. ASSIGN NEW OWNERS
            # new_owner = new_obj_owner[j].
            # assignment[new_owner] = j.
            # Indices: new_obj_owner (B, M).
            # Src: arange(M).
            # Condition: changed_mask is True.
            
            target_new_agents = new_obj_owner.clamp(min=0)
            current_vals_new = assignment.gather(1, target_new_agents)
            
            # Obj IDs to assign
            obj_ids = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
            
            assign_vals = torch.where(changed_mask, obj_ids, current_vals_new)
            
            assignment.scatter_(1, target_new_agents, assign_vals)
            
            # 4. Commit State
            prices = new_prices
            obj_owner = new_obj_owner
            
            # check convergence?
            # if (assignment != -1).all(): break
            # But generic auction usually runs fixed iter or eps scaling.
            # Allow pure fixed iter for compilation simplicity (removes graph break).
            
        return assignment
    
    return torch.compile(outer, dynamic=False, fullgraph=True, options=options)

# --- Public Instances ---
sinkhorn_compiled = CompiledKernelDispatcher(_create_sinkhorn)
dual_ascent_compiled = CompiledKernelDispatcher(_create_dual_ascent)
auction_compiled = CompiledKernelDispatcher(_create_auction)

