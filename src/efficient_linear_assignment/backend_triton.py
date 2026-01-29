import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def bidding_kernel(
    benefits_ptr, prices_ptr, 
    bids_ptr, best_obj_indices_ptr,
    increments_ptr,
    N, M,
    epsilon: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr
):
    """
    Computes bids for a single agent (row) across all objects (cols).
    Grid: (N, ) - One program per agent? 
    Or tiled? If M is large, we need tiling.
    For simplicity, assume M fits in block or loop.
    
    If M is very large, efficient reduction is needed.
    Let's handle the case where M splits across blocks? 
    No, Auction usually implies dense small-medium M or requires complex logic.
    We'll implement a row-per-program kernel with loop over M for now.
    """
    # Agent index
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    # Pointers
    row_start_ptr = benefits_ptr + pid * M
    prices_base = prices_ptr
    
    # Track top 2 values
    best_val = -float('inf')
    second_val = -float('inf')
    best_idx = -1
    
    # Loop over M in chunks
    for off in range(0, M, BLOCK_SIZE_M):
        cols = off + tl.arange(0, BLOCK_SIZE_M)
        mask = cols < M
        
        # Load benefits and prices
        # benefits[pid, cols]
        ben = tl.load(row_start_ptr + cols, mask=mask, other=-float('inf'))
        # prices[cols] (Prices are shared across agents, (M))
        # Note: If batch size > 1, prices ptr needs offset! 
        # The kernel needs a Batch dim if we support batching.
        # Let's start with single batch support or handle batch offset in caller.
        pr = tl.load(prices_base + cols, mask=mask, other=float('inf')) 
        
        # value = benefit - price
        # Maximize Benefit = (Benefit - Price)
        val = ben - pr
        
        # Local Top 2 Reduction
        # This is hard within a block efficiently without standard reduc utils.
        # But here we are within a thread (program) that owns the loop?
        # WAIT. Structure:
        # Standard Triton GEMM style: 1 block = tile of rows x tile of cols.
        # If we assign one row per thread, we serialize the search over M. Too slow.
        # We need block reduction. 
        
        # Structure: 1 Block per Agent (Row).
        # Threads cooperate to reduce the row.
        pass

    # RE-THINKING:
    # A simple approach for Triton:
    # Kernel 'find_top2':
    # Input: Benefits (B, N, M), Prices (B, M)
    # Output: BestIdx (B, N), Increment (B, N)
    
    # Parallelism: (B, N) blocks?
    # Inside block: Reduction over M.
    pass

@triton.jit
def find_top2_kernel(
    benefits_ptr, prices_ptr,
    best_vals_ptr, second_vals_ptr, best_inds_ptr,
    stride_b_n, stride_b_m, # Stride for benefits
    stride_p_b, stride_p_m, # Stride for prices
    M,
    BLOCK_SIZE: tl.constexpr
):
    # Map program ID to (Batch, Agent)
    # Grid is 1D or 2D? (B*N).
    pid = tl.program_id(0)
    # We can decompose pid if needed, or pass offsets.
    
    # Assume flat grid for (B * N) tasks.
    # Current task: Agent 'pid'.
    # We need to locate this agent's row in Benefits.
    
    # Global offset:
    # The caller treats (B, N) as a single flat dimension 'RowTotal'.
    # But prices are (B, M), so we repeat prices every N agents?
    # We need to know which Batch 'b' we are in.
    # b = pid // N
    # agent_idx = pid % N
    
    # Since N is potentially symbolic or passed in, division is expensive?
    # Better to launch with 2D grid (N, B)? 
    # Yes: program_id(0) = agent_idx, program_id(1) = batch_idx
    
    agent_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    # Pointers
    # Benefits: [batch, agent, :]
    ben_row_start = benefits_ptr + batch_idx * stride_b_n + agent_idx * stride_b_m
    
    # Prices: [batch, :]
    price_row_start = prices_ptr + batch_idx * stride_p_b
    
    # Accumulators for this block
    # We use shared memory implicit in reductions?
    # We iterate tile by tile.
    
    max1_val = -float('inf')
    max2_val = -float('inf')
    max1_idx = -1
    
    # Iterate over M cols
    for off in range(0, M, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < M
        
        # Load
        # benefits[b, n, cols]
        # Stride for last dim is assumed 1? 
        # stride_b_m refers to stride of dim 1? No, usually dim 2 stride is 1.
        # Pointer arithmetic assumes linear memory.
        b_vals = tl.load(ben_row_start + cols, mask=mask, other=-float('inf'))
        p_vals = tl.load(price_row_start + cols, mask=mask, other=float('inf'))
        
        # Net value
        vals = b_vals - p_vals
        
        # --- Reduction ---
        # Find top 2 in this chunk
        # Note: Triton doesn't have a direct 'topk' instruction for vectors usually.
        # We can do `max` easily.
        
        # This implementation requires efficient intra-block reduction.
        # Doing it purely in register loop per thread is maybe okay if BLOCK_SIZE is large enough to cover M?
        # If BLOCK_SIZE == M (padded), then we just reduce.
        
        # Currently, let's assume one thread handles the row loop?
        # NO. Triton threads run in parallel.
        # If we use 1 block per row, threads split the columns.
        pass

# Fallback to a simplified kernel structure for readability in this first iteration.
# We will use a "Softmax-style" reduction approach.
# Each block handles one row (Agent).
# Threads load chunks of M, compute local top2, then reduce across threads.

@triton.jit
def auction_bid_kernel(
    benefits_ptr, # (B, N, M)
    prices_ptr,   # (B, M)
    assignment_ptr, # (B, N) - to mask assigned
    best_idx_ptr, # (B, N) Out
    increment_ptr, # (B, N) Out
    epsilon,
    B, N, M,
    stride_bn, stride_bm,
    stride_bp,
    BLOCK_SIZE: tl.constexpr
):
    # 2D Grid: (N, B)
    # pid_0 = agent_idx (0..N-1)
    # pid_1 = batch_idx (0..B-1)
    
    row_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    # Check assignment status
    # If assigned != -1, skip?
    # assignment pointer offset: batch * N + row
    assign_ptr_loc = assignment_ptr + batch_idx * N + row_idx
    # current_assign = tl.load(assign_ptr_loc)
    # if current_assign != -1:
    #     return # Skip bidding
    
    # Actually, we compute for all and mask later or skip here.
    # Skipping here saves power/compute.
    current_assign = tl.load(assign_ptr_loc)
    if current_assign != -1:
        # Write dummy
        tl.store(best_idx_ptr + batch_idx * N + row_idx, -1)
        tl.store(increment_ptr + batch_idx * N + row_idx, 0.0)
        return

    # Pointers
    ben_ptr_base = benefits_ptr + batch_idx * stride_bn + row_idx * stride_bm
    price_ptr_base = prices_ptr + batch_idx * stride_bp
    
    # Reductions
    # Algorithm:
    # Thread k loads elements k, k+BLOCK, etc.
    # Computes thread-local top2.
    # Then cross-thread reduction.
    
    tid = tl.arange(0, BLOCK_SIZE)
    
    # Local State
    thread_max1 = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    thread_max2 = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    thread_idx1 = tl.full((BLOCK_SIZE,), -1, dtype=tl.int64)
    
    # Loop over columns with stride BLOCK_SIZE
    for off in range(0, M, BLOCK_SIZE):
        idx = (off + tid).to(tl.int64)
        mask = idx < M
        
        # Load
        b_val = tl.load(ben_ptr_base + idx, mask=mask, other=-float('inf')).to(tl.float32)
        p_val = tl.load(price_ptr_base + idx, mask=mask, other=float('inf')).to(tl.float32)
        
        val = b_val - p_val
        
        # Update thread local state
        # Compare val with thread_max1, thread_max2
        # This conditional logic is fine in Triton
        is_new_max = val > thread_max1
        
        # If new max
        # max2 becomes max1, max1 becomes val
        thread_max2 = tl.where(is_new_max, thread_max1, tl.maximum(thread_max2, val))
        thread_max1 = tl.where(is_new_max, val, thread_max1)
        thread_idx1 = tl.where(is_new_max, idx, thread_idx1)
    
    # Reduce across threads
    # We need to collect top 2 across all threads.
    # Standard triton.reduce only reduces one value.
    # We can try to serialize or use shared memory.
    
    # Simplified: Reduce max1 first.
    # global_max1 = tl.max(thread_max1) -- this reduces locally? No.
    # tl.reduce(input, axis, op)
    
    # Let's pack (val, idx) into int64 for atomic max? No, float values.
    
    # Use cross-lane ops?
    # Not easily exposed.
    
    # Fallback: Loop? No.
    # We can perform reduction in shared memory if needed, but Triton hides this.
    # `tl.reduce` returns the reduction of the tensor.
    # But `thread_max1` is a register/tensor of size BLOCK_SIZE.
    # We can reduce it.
    
    # 1. Find Global Max 1
    # Note: argmax reduction needed.
    # We can pack value and index? 
    # Or just reduce value, then broadcast, then find who had it?
    
    # Max value
    block_max1 = tl.max(thread_max1, axis=0) # Scalar
    
    # Identify who had it
    # mask = (thread_max1 == block_max1)
    # winner_idx = tl.where(mask, thread_idx1, -1)
    # block_idx1 = tl.max(winner_idx, axis=0)
    
    # 2. Find Global Max 2
    # Mask out the winner
    # val_for_second = tl.where(thread_idx1 == block_idx1, thread_max2, thread_max1)
    # block_max2 = tl.max(val_for_second, axis=0)
    
    # Store
    # best_idx_ptr[batch, row] = block_idx1
    # increment = block_max1 - block_max2 + epsilon
    
    # Note: `tl.max` on a register tensor reduces it.
    
    g_max1 = tl.max(thread_max1, axis=0)
    
    # Find index of max
    # Note: ties broken arbitrarily
    match_mask = (thread_max1 == g_max1)
    # Get index (global col index)
    # thread_idx1 holds the global column index of the local max
    
    # We need to pick one index if multiple threads match.
    # max of index is a simple tie break.
    g_idx1 = tl.max(tl.where(match_mask, thread_idx1, -1), axis=0)
    
    # compute second max
    # If a thread held the max, its *contribution* to second max is its `thread_max2`.
    # If a thread did NOT hold the max, its contribution is its `thread_max1`.
    
    # Warning: unique logic.
    # If tid 0 has [10, 5], max1=10.
    # If tid 1 has [8, 7], max1=8.
    # Global max = 10.
    # Candidates for second:
    # tid 0 provides 5.
    # tid 1 provides 8.
    # max(5, 8) = 8.
    
    # So:
    # candidate_val = where(thread_id_matches_global, thread_max2, thread_max1)
    # This assumes thread_id_matches_global implies THIS thread provided the global max.
    # What if index check?
    # Checking `thread_idx1 == g_idx1` is correct.
    
    is_winner = (thread_idx1 == g_idx1)
    candidates = tl.where(is_winner, thread_max2, thread_max1)
    g_max2 = tl.max(candidates, axis=0)
    
    # Write output
    offset = batch_idx * N + row_idx
    tl.store(best_idx_ptr + offset, g_idx1)
    tl.store(increment_ptr + offset, g_max1 - g_max2 + epsilon)



# ---------------------------------------------------------------------------
# Helper: Pack/Unpack Utilities
# ---------------------------------------------------------------------------
# We pack (Price: float32, AgentID: int32) into uint64.
# Float32 is reinterpret_cast to int32 (or uint32) for storage.
# We assume Price is non-negative for direct comparison?
# Check: Benefits can be negative. Prices can be negative?
# Auction: Prices start at 0. eps > 0. Prices increase.
# So Prices >= 0.
# IEEE-754 positive floats preserve order when interpreted as uint32.
# So we can just cast float->uint32, shift, and OR with agent_id.

@triton.jit
def pack_bid(price: tl.float32, agent_id: tl.int32):
    # Reinterpret float bits as uint32
    # Note: Triton's `to` with bitcast behavior?
    # tl.bitcast needs same size.
    # float32 -> uint32
    price_bits = price.to(tl.uint32, bitcast=True)
    # Pack: High 32 bits = Price, Low 32 bits = AgentID
    # Cast to uint64
    high = price_bits.to(tl.uint64) << 32
    low = agent_id.to(tl.uint64) & 0xFFFFFFFF
    return high | low

@triton.jit
def unpack_bid(packed: tl.uint64):
    high = packed >> 32
    low = packed & 0xFFFFFFFF
    # Recover float
    price = high.to(tl.uint32).to(tl.float32, bitcast=True)
    agent_id = low.to(tl.int32)
    return price, agent_id

# ---------------------------------------------------------------------------
# Update Phase Kernels
# ---------------------------------------------------------------------------

@triton.jit
def auction_scatter_kernel(
    best_idx_ptr,      # (B, N) int64
    increments_ptr,    # (B, N) float32
    prices_ptr,        # (B, M) float32
    proposals_ptr,     # (B, M) uint64 (Zeroed before call)
    assignment_ptr,    # (B, N) int64
    N: tl.constexpr,   # Number of agents
    M: tl.constexpr,   # Number of objects
    stride_bn,         # Stride for best_idx, increments, assignment (B, N)
    stride_bm,         # Stride for prices, proposals (B, M)
    BLOCK_SIZE: tl.constexpr
):
    """
    Each thread handles one agent (or block handles chunk of agents).
    We use a flat 1D grid of size (B * N).
    """
    pid = tl.program_id(0)
    # Map to Batch, Agent
    # Assuming (B*N) grid with block size 1?
    # No, typically Triton kernels use block_size threads.
    # Let's assume Grid = (B * N + BLOCK - 1) // BLOCK.
    # Each thread handles one agent.
    
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate Batch and Agent ID
    # This requires division, which can be slow, but for 1D launch it's necessary
    # effectively:
    # batch_id = offset // N
    # agent_id = offset % N
    # To avoid division in loop, we can assume caller launches 2D grid?
    # Let's stick to 2D grid launch: (N_blocks, B)
    # where N_blocks covers N. 
    # But for atomic scatter, Agents are independent.
    
    # Let's use 1D launch logic with masking.
    
    batch_n_idx = offset
    mask = batch_n_idx < (stride_bn * stride_bm) # invalid check? 
    # Actually simpler: Total items = B * N.
    total_agents = grid_dims_x = tl.num_programs(0) * BLOCK_SIZE # Not available directly?
    
    # Let's use the provided inputs B, N to decode.
    # We assume 'stride_bn' is actually the stride for the Batch dimension (i.e., N).
    # So index i corresponds to:
    # b = i // N
    # n = i % N
    
    # Wait, simple pointer arithmetic:
    # Inputs are flat pointers + offset.
    # We just need to ensure we access the right `prices` and `proposals` row.
    
    # Global Agent Index
    global_agent_id = offset
    
    # Check bounds (assume N total is passed as limit, or we compute from shape?)
    # We'll pass `total_agents` as uniform limit?
    # Let's rely on mask.
    # The Grid should cover B*N.
    # We need B and N to calculate correct price offset (Batch).
    
    # Derived Batch/Row
    # b = global_agent_id // N
    # n = global_agent_id % N
    # This is expensive. 
    # Alternative: 2D Grid (N_blocks, B)
    # pid_x = blockIdx.x, pid_y = blockIdx.y
    # agent_start = pid_x * BLOCK_SIZE
    
    # Let's use 2D Grid approach for cleaner indexing:
    # X dim: Agents (tiled)
    # Y dim: Batch
    
    pass

# Refined Scatter Kernel with 2D Grid
@triton.jit
def auction_scatter_kernel_2d(
    best_idx_ptr,      # (B, N)
    increments_ptr,    # (B, N)
    prices_ptr,        # (B, M)
    proposals_ptr,     # (B, M)
    assignment_ptr,    # (B, N)
    N, M,
    BLOCK_SIZE: tl.constexpr
):
    # Grid: (ceil(N/BLOCK), B)
    pid_x = tl.program_id(0) # Block ID along N
    pid_y = tl.program_id(1) # Batch ID
    
    # Agent indices for this block
    agent_ids = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = agent_ids < N
    
    # Base pointers for this batch
    # Arrays are (B, N) -> Offset = bid * N
    offset_bn = pid_y * N + agent_ids
    
    # Assignment Check
    assign_ptr = assignment_ptr + offset_bn
    status = tl.load(assign_ptr, mask=mask, other=0) # 0 ok, checks later
    
    # Filter: Only proceed if unassigned (-1) based on mask
    active_mask = mask & (status == -1)
    
    # Load Best Object Index
    best_idx_ptr_loc = best_idx_ptr + offset_bn
    # If mask is false/unassigned, load 0 safely to avoid OOB
    target_obj = tl.load(best_idx_ptr_loc, mask=active_mask, other=0)
    
    # Load Increment
    inc_ptr = increments_ptr + offset_bn
    increment = tl.load(inc_ptr, mask=active_mask, other=0.0).to(tl.float32)
    
    # Load Current Price of Target
    # Prices (B, M) -> Offset = bid * M + target_obj
    # Since 'target_obj' is a tensor of indices, we use indirect addressing
    price_base_ptr = prices_ptr + pid_y * M
    current_price = tl.load(price_base_ptr + target_obj, mask=active_mask, other=0.0).to(tl.float32)
    
    # Compute Bid
    new_bid = current_price + increment
    
    # Verify validity (e.g., target_obj != -1)
    # existing best_idx might be -1 if no valid options
    valid_bid = active_mask & (target_obj != -1)
    
    # Pack
    # Only pack if valid.
    # Note: we need casting. 'agent_ids' are the intra-batch indices (0..N-1).
    packed = pack_bid(new_bid, agent_ids.to(tl.int32))
    
    # Atomic Max into Proposals
    # Proposals (B, M)
    prop_base_ptr = proposals_ptr + pid_y * M
    write_ptr = prop_base_ptr + target_obj
    
    # execute atomic only for valid lanes
    tl.atomic_max(write_ptr, packed, mask=valid_bid)


@triton.jit
def auction_resolve_kernel_2d(
    best_idx_ptr,      # (B, N)
    assignment_ptr,    # (B, N)
    prices_ptr,        # (B, M)
    proposals_ptr,     # (B, M)
    owner_ptr,         # (B, M)
    N, M,
    BLOCK_SIZE: tl.constexpr
):
    # Grid: (ceil(N/BLOCK), B)
    # We iterate over ACENTS again to check if they won.
    
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    agent_ids = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = agent_ids < N
    
    offset_bn = pid_y * N + agent_ids
    
    # Check Unassigned
    status = tl.load(assignment_ptr + offset_bn, mask=mask, other=0)
    active_mask = mask & (status == -1)
    
    # Load what I bid on
    target_obj = tl.load(best_idx_ptr + offset_bn, mask=active_mask, other=0)
    
    # Read Result from Proposals
    prop_base_ptr = proposals_ptr + pid_y * M
    # Check bounds of target_obj just in case
    # If target_obj == -1, ignore
    valid_target = active_mask & (target_obj != -1)
    
    winning_packed = tl.load(prop_base_ptr + target_obj, mask=valid_target, other=0)
    
    # Unpack
    win_price, win_agent = unpack_bid(winning_packed)
    
    # Check if I am the winner
    # agent_ids is int64 range? Cast to int32 for comparison
    i_won = valid_target & (win_agent == agent_ids.to(tl.int32))
    
    # If I won:
    # 1. Update assignment[me] = target_obj
    # 2. Update prices[target_obj] = win_price
    # 3. Update owner[target_obj] = me
    # 4. Unassign old owner (Read-Modify-Write needs care?)
    
    # Handling Old Owner:
    # owner_ptr has old owner.
    # Multiple winners (different OBJS) might try to unassign different OLD owners.
    # No conflict on 'assignment' array since 'old_owner' indices are unique?
    # Yes, one object has one owner.
    
    if tl.max(i_won.to(tl.int32), axis=0): # Optimization: any winner in block?
        # Since we can't branch easily on specific lanes, we just execute masked stores.
        
        # 1. Mark Me Assigned
        tl.store(assignment_ptr + offset_bn, target_obj, mask=i_won)
        
        # 2. Update Price
        price_ptr_loc = prices_ptr + pid_y * M + target_obj
        tl.store(price_ptr_loc, win_price, mask=i_won)
        
        # 3. Handle Old Owner
        owner_ptr_loc = owner_ptr + pid_y * M + target_obj
        old_owner = tl.load(owner_ptr_loc, mask=i_won, other=-1)
        
        # If old_owner != -1, unassign them
        # old_owner is int64
        # Pointer to assignment[old_owner]
        # Base assignment ptr for batch: assignment_ptr + pid_y * N
        old_assign_ptr = assignment_ptr + pid_y * N + old_owner
        
        has_old = i_won & (old_owner != -1)
        tl.store(old_assign_ptr, -1, mask=has_old)
        
        # 4. Set New Owner
        tl.store(owner_ptr_loc, agent_ids, mask=i_won)




class AuctionTriton:
    def __init__(self, epsilon: float = 1e-2, max_iter: int = 1000):
        self.epsilon = epsilon
        self.max_iter = max_iter

    def solve(self, cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Implementation of solve using the kernels
        # Similar logic to AuctionTorch but calls kernel for bidding
        
        # Keep input dtype (e.g. F16) -> Kernel casts to F32
        benefits = -cost_matrix
        
        B, N, M = benefits.shape
        prices = torch.zeros((B, M), device=benefits.device, dtype=torch.float32)
        assignment = torch.full((B, N), -1, device=benefits.device, dtype=torch.long)
        
        # Pre-allocate buffers
        best_idx = torch.full((B, N), -1, device=benefits.device, dtype=torch.long)
        increments = torch.zeros((B, N), device=benefits.device, dtype=torch.float32)
        
        # Buffers for Phase 1 Update
        # Proposals: Packed (Price, AgentID)
        # We need int64 view of uint64 for PyTorch creation? 
        # PyTorch doesn't strictly support uint64. int64 is fine for storage.
        proposals = torch.zeros((B, M), device=benefits.device, dtype=torch.int64)
        
        # Object Owners: Tracks who currently owns obj j
        object_owners = torch.full((B, M), -1, device=benefits.device, dtype=torch.long)
        
        # Grid config
        BLOCK_SIZE = triton.next_power_of_2(M)
        if BLOCK_SIZE < 128: BLOCK_SIZE = 128
        if BLOCK_SIZE > 1024: BLOCK_SIZE = 1024
        
        # Block size for Agent Loops (Scatter/Resolve)
        # We process agents (N).
        AGENT_BLOCK_SIZE = 128
        num_agent_blocks = (N + AGENT_BLOCK_SIZE - 1) // AGENT_BLOCK_SIZE
        
        # 2D Grid for Agent Kernels: (num_blocks, B)
        # Note: Triton grid arg is tuple.
        # 2D Grid for Agent Kernels: (num_blocks, B)
        # Note: Triton grid arg is tuple.
        agent_grid = (num_agent_blocks, B)
        
        # Grid for Bidding
        grid = (N, B)
        
        
        # 4. Burst Execution with CUDA Graphs
        # We process 'BURST_SIZE' iterations per CPU sync.
        BURST_SIZE = 10
        
        # Disable CUDA Graph manual capture if we are already compiling (Inductor manages graph)
        is_compiling = False
        if hasattr(torch, "compiler") and torch.compiler.is_compiling():
            is_compiling = True
            
        use_cuda_graph = True and not is_compiling
        
        # Define the burst sequence
        def run_check_step():
             # Single step logic
             # Bidding
             auction_bid_kernel[grid](
                 benefits, prices, assignment,
                 best_idx, increments,
                 self.epsilon,
                 B, N, M,
                 benefits.stride(0), benefits.stride(1),
                 prices.stride(0),
                 BLOCK_SIZE=BLOCK_SIZE
             )
             
             proposals.zero_()
             
             auction_scatter_kernel_2d[agent_grid](
                 best_idx, increments, prices, proposals, assignment,
                 N, M,
                 BLOCK_SIZE=AGENT_BLOCK_SIZE
             )
             
             auction_resolve_kernel_2d[agent_grid](
                 best_idx, assignment, prices, proposals, object_owners,
                 N, M,
                 BLOCK_SIZE=AGENT_BLOCK_SIZE
             )

        if use_cuda_graph:
            # Warmup
            # Run once to ensure allocations/compilation?
            # Triton kernels JIT compile on first launch.
            # We already ran similar kernels? No, let's run one burst.
            for _ in range(3):
                run_check_step()
            
            # Reset solver state for actual run?
            # No, solving is iterative. Warmup just progressed the solution.
            # But we want to capture the *sequence*.
            
            # Capture
            g = torch.cuda.CUDAGraph()
            
            # We need to capture the *exact* sequence of launches.
            # Torch CUDAGraph requires side-effects to be on captured tensors.
            # Our tensors (assignment, prices, etc.) are fixed buffers.
            
            with torch.cuda.graph(g):
                for _ in range(BURST_SIZE):
                    run_check_step()
            
            # Main Loop
            for i in range(0, self.max_iter, BURST_SIZE):
                # Check Convergence (CPU Sync)
                # We check *before* running the burst? or after?
                # Usually check after previous burst using the result.
                
                # Note: unassigned_count check involves CPU sync.
                unassigned_count = (assignment == -1).sum().item()
                if unassigned_count == 0:
                    break
                
                if i > 0: # Did we overshoot? No, kernels are safe.
                   pass
                
                g.replay()
                
        else:
            # Fallback manual loop
            for i in range(self.max_iter):
                 unassigned_count = (assignment == -1).sum().item()
                 if unassigned_count == 0:
                     break
                 run_check_step()
            
        return assignment, prices
