
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def update_best(
    best_val, best_idx,
    second_val,
    cand_val, cand_idx
):
    """
    Updates (best, second) with a new candidate.
    """
    if cand_val > best_val:
        second_val = best_val
        best_val = cand_val
        best_idx = cand_idx
    elif cand_val > second_val:
        second_val = cand_val
        
    return best_val, best_idx, second_val

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_M': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 256}, num_warps=8, num_stages=3),
    ],
    key=['B', 'N', 'M', 'D'],
)
@triton.jit
def auction_bid_implicit_kernel(
    # Pointers
    Q_ptr,      # (B, N, D)
    K_ptr,      # (B, M, D)
    Prices_ptr, # (B, M)
    
    # Outputs
    BestIdx_ptr,   # (B, N)
    Increments_ptr,# (B, N)
    
    # Shapes
    B, N, M, D,
    
    # Strides
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_km, stride_kd,
    stride_pb, stride_pm,
    
    # Constants
    epsilon,
    
    # Block sizes
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Implicit Auction Bidding Phase.
    Computes Cost[i, j] = dot(Q[i], K[j]) - Prices[j] on the fly.
    Finds top-2 values for each row i.
    """
    # Program ID
    pid_b = tl.program_id(1) # Batch
    pid_n = tl.program_id(0) # Block of Rows
    
    # -----------------------------------------------------------
    # 1. Pointers for Q (Query/Source)
    # -----------------------------------------------------------
    # Row indices for this block
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Mask for valid rows
    mask_n = offs_n < N
    
    # Q ptr: Base + batch + row*stride + col*stride
    # We load D dimensions. BLOCK_D must cover D (or loop, but D usually small 64-128).
    # Assume D <= BLOCK_D for "Flash" style simple implementation usually.
    # Or loop over D? dot(Q, K.T) accumulates. 
    # Let's support looping over D if needed, but standard is D=head_dim.
    
    offs_d = tl.arange(0, BLOCK_D)
    
    # Load Q block: (BLOCK_N, BLOCK_D)
    # Ptr: BaseQ + (pid_b * stride_qb) + (offs_n[:, None] * stride_qn) + (offs_d[None, :] * stride_qd)
    Q_load_ptr = Q_ptr + (pid_b * stride_qb) + \
                 (offs_n[:, None] * stride_qn) + \
                 (offs_d[None, :] * stride_qd)
                 
    # Mask Q: row in range, d in range
    mask_q = mask_n[:, None] & (offs_d[None, :] < D)
    
    # Load Q. Use float16/bfloat16 for Tensor Cores!
    # Input should be cast before.
    q = tl.load(Q_load_ptr, mask=mask_q, other=0.0)
    
    # -----------------------------------------------------------
    # 2. Accumulators for Top-2
    # -----------------------------------------------------------
    # Per row in BLOCK_N
    best_val = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    second_val = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    best_idx = tl.full([BLOCK_N], -1, dtype=tl.int32)
    
    # -----------------------------------------------------------
    # 3. Loop over K (Target/Objects) in chunks of BLOCK_M
    # -----------------------------------------------------------
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        
        # Load K block: (BLOCK_M, BLOCK_D)
        # Note: We need K transposed for dot: (D, BLOCK_M)? 
        # tl.dot(A, B). A=(M, K), B=(K, N).
        # Q=(BLOCK_N, D). We need K_trans=(D, BLOCK_M).
        # We load K normally (BLOCK_M, D) then trans?
        # Ideally load K_trans directly if memory layout allows?
        # Usually K is row-major. 
        # Load (BLOCK_M, D) -> trans -> (D, BLOCK_M).
        
        K_load_ptr = K_ptr + (pid_b * stride_kb) + \
                     (offs_m[:, None] * stride_km) + \
                     (offs_d[None, :] * stride_kd)
                     
        mask_k = mask_m[:, None] & (offs_d[None, :] < D)
        k_chunk = tl.load(K_load_ptr, mask=mask_k, other=0.0)
        
        # Transpose K for dot
        k_trans = tl.trans(k_chunk) # (D, BLOCK_M)
        
        # -------------------------------------------------------
        # Compute Scores = Q @ K.T
        # -------------------------------------------------------
        # Q=(N, D), K.T=(D, M) -> (N, M)
        # Accumulator precision float32
        scores = tl.dot(q, k_trans)
        
        # -------------------------------------------------------
        # Subtract Prices
        # -------------------------------------------------------
        # Load Prices: (BLOCK_M)
        P_load_ptr = Prices_ptr + (pid_b * stride_pb) + offs_m
        prices = tl.load(P_load_ptr, mask=mask_m, other=float('inf'))
        
        # Broadcast prices to (BLOCK_N, BLOCK_M)
        # scores = scores - prices[None, :]
        scores = scores - prices[None, :]
        
        # Mask out-of-bounds columns (M)
        # Set to -inf
        scores = tl.where(mask_m[None, :], scores, -float('inf'))
        
        # -------------------------------------------------------
        # Row-wise Reduction (Top-2) within this chunk
        # -------------------------------------------------------
        # We have a (BLOCK_N, BLOCK_M) matrix in registers.
        # We need top-2 for each row.
        # Standard generic max/argmax is easy. Second max is harder.
        
        # 1. First Max
        current_best_val, current_best_idx_local = tl.max(scores, axis=1, return_indices=True)
        # Convert local index to global
        current_best_idx = start_m + current_best_idx_local
        
        # 2. Second Max
        # Mask out the best
        # Need to reconstruct mask. 
        # tricky in triton without mutable arrays.
        # broadcast best_idx_local back to (N, M) and compare?
        # indices = tl.arange(0, BLOCK_M)[None, :]
        # mask_not_best = indices != current_best_idx_local[:, None]
        # scores_masked = tl.where(mask_not_best, scores, -float('inf'))
        # current_second_val = tl.max(scores_masked, axis=1)
        
        offs_m_local = tl.arange(0, BLOCK_M)
        mask_not_best = offs_m_local[None, :] != current_best_idx_local[:, None]
        scores_no_best = tl.where(mask_not_best, scores, -float('inf'))
        current_second_val = tl.max(scores_no_best, axis=1)
        
        # -------------------------------------------------------
        # Update Global Accumulators
        # -------------------------------------------------------
        # Compare (current_best, current_second) with (best, second)
        
        # Case 1: current_best > best
        #   new_best = current_best
        #   new_second = max(best, current_second)
        # Case 2: current_best <= best
        #   new_best = best
        #   new_second = max(second, current_best)
        
        is_new_best = current_best_val > best_val
        
        # Prepare "evicted" best (becomes candidate for second)
        old_best_val = best_val
        
        # Update Best
        best_val = tl.where(is_new_best, current_best_val, best_val)
        best_idx = tl.where(is_new_best, current_best_idx, best_idx)
        
        # Update Second
        # Candidates for second: 
        # If is_new_best: max(old_best_val, current_second_val)
        # Else:           max(second_val, current_best_val)
        
        cand_for_second = tl.where(is_new_best, old_best_val, current_best_val)
        # Is current_second_val relevant if is_new_best? Yes.
        # But we simplified: new_second = max(second_val, cand_for_second) ?
        
        # Wait, fully correct logic:
        # Four values: old_b, old_s, curr_b, curr_s.
        # But we know old_b >= old_s, curr_b >= curr_s.
        # Max is max(old_b, curr_b).
        # Second is max(min(old_b, curr_b), old_s, curr_s).
        # Actually simplifies to max(min(old_b, curr_b), max(old_s, curr_s)) if disjoint?
        # No, simpler:
        # v1, i1 = old_b, old_i
        # v2 = old_s
        # u1, j1 = curr_b, curr_j
        # u2 = curr_s
        
        # if u1 > v1:
        #    best = u1
        #    second = max(v1, u2) (since v1 >= v2, u1 >= u2)
        # else:
        #    best = v1
        #    second = max(v2, u1)
        
        # Logic:
        # new_second_if_swap = tl.maximum(old_best_val, current_second_val)
        # new_second_no_swap = tl.maximum(second_val, current_best_val)
        # second_val = tl.where(is_new_best, new_second_if_swap, new_second_no_swap)

        new_sec_swap = tl.maximum(old_best_val, current_second_val)
        new_sec_keep = tl.maximum(second_val, current_best_val)
        
        second_val = tl.where(is_new_best, new_sec_swap, new_sec_keep)
        
    # End Loop over K

    # -----------------------------------------------------------
    # 4. Store Results
    # -----------------------------------------------------------
    # Increment = Best - Second + Epsilon
    increment = best_val - second_val + epsilon
    
    # Store BestIdx
    idx_ptr = BestIdx_ptr + (pid_b * N) + offs_n
    tl.store(idx_ptr, best_idx, mask=mask_n)
    
    # Store Increment
    inc_ptr = Increments_ptr + (pid_b * N) + offs_n
    tl.store(inc_ptr, increment, mask=mask_n)

class AuctionImplicitTriton:
    def __init__(self, epsilon=1.0, max_iter=500):
        self.epsilon = epsilon
        self.max_iter = max_iter
        
    def solve(self, Q: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves Linear Assignment for Cost = Q @ K.T.
        Args:
            Q: (B, N, D) or (N, D)
            K: (B, M, D) or (M, D)
        """
        if Q.ndim == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
            
        B, N, D = Q.shape
        _, M, _ = K.shape
        
        device = Q.device
        
        # State
        assignment = torch.full((B, N), -1, device=device, dtype=torch.long)
        prices = torch.zeros((B, M), device=device, dtype=torch.float32)
        
        # Check inputs likely FP16/BF16
        # Kernel handles computation in float32 (dot output) but loading in input dtype.
        
        # Grid
        # Dynamic based on autotuned BLOCK_N
        grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), B)
        
        BLOCK_D = triton.next_power_of_2(D)

        # Pre-allocate outputs for kernel
        best_idx = torch.full((B, N), -1, device=device, dtype=torch.int32)
        increments = torch.zeros((B, N), device=device, dtype=torch.float32)
        
        # Bid-Resolve Loop
        
        from efficient_linear_assignment.auction.triton_backend import auction_scatter_kernel_2d, auction_resolve_kernel_2d
        
        # Proposals: (B, M) -- Packed (bid, agent)
        proposals = torch.zeros((B, M), device=device, dtype=torch.int64)
        owners = torch.full((B, M), -1, device=device, dtype=torch.long) 
        
        # NOTE: Scatter/Resolve need BLOCK_N? 
        # We can fix their block size or use a heuristic.
        # Let's use 128 for them as they are element-wise on Agents/Objects.
        AGENT_BLOCK = 128
        agent_grid = (triton.cdiv(N, AGENT_BLOCK), B)
        
        # Generator Check
        is_generator = hasattr(self.epsilon, '__next__')
        current_epsilon = self.epsilon
        if is_generator:
             try:
                current_epsilon = next(self.epsilon)
             except StopIteration:
                pass

        for i in range(self.max_iter):
            # Update Epsilon if generator
            if is_generator and i > 0:
                try:
                    current_epsilon = next(self.epsilon)
                except StopIteration:
                    pass

            # 1. Bid (Implicit)
            auction_bid_implicit_kernel[grid](
                Q, K, prices,
                best_idx, increments,
                B, N, M, D,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                prices.stride(0), prices.stride(1),
                current_epsilon,
                BLOCK_D=BLOCK_D
            )
            
            # Zero proposals
            proposals.zero_()
            
            # 2. Scatter (Standard)
            # Re-use from triton_backend or implement?
            # Existing arguments: best_idx_ptr, increments_ptr, prices_ptr, proposals_ptr, assignment_ptr, N, M
            auction_scatter_kernel_2d[agent_grid](
                best_idx, increments, prices, proposals, assignment,
                N, M,
                BLOCK_SIZE=AGENT_BLOCK
            )
            
            # 3. Resolve (Standard)
            # Existing arguments: best_idx_ptr, assignment_ptr, prices_ptr, proposals_ptr, owner_ptr, N, M
            # Grid?
            auction_resolve_kernel_2d[agent_grid](
                 best_idx, assignment, prices, proposals, owners,
                 N, M,
                 BLOCK_SIZE=AGENT_BLOCK
            )
            
            # Convergence check?
            # Only if we track unassigned count.
            # Lazy check: if iter % 100 == 0: check unassigned?
            
        return assignment, prices
