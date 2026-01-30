import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['B', 'N', 'M'],
)
@triton.jit
def _dual_ascent_update_kernel(
    C_ptr,           # (B, N, M)
    pot_in_ptr,      # (B, InDim) (g/beta for row update)
    pot_out_ptr,     # (B, OutDim) (f/alpha for row update)
    marginal_ptr,    # (B, OutDim)
    B, N, M,
    stride_cb, stride_cn, stride_cm,
    stride_pin_b, stride_pin_d,
    stride_pout_b, stride_pout_d,
    stride_marg_b, stride_marg_d,
    epsilon,
    is_row_update: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Newton Step:
    # T = alpha + beta - C
    # Active = T > 0
    # CurrentSum = Sum(T * Active) / eps
    # Grad = Count(Active) / eps
    # Delta = (Marginal - CurrentSum) / ClampedGrad
    # AlphaNew = Alpha + Delta
    
    pid = tl.program_id(0)
    
    if is_row_update:
        num_out = N
        num_in = M
    else:
        num_out = M
        num_in = N
        
    batch_idx = pid // num_out
    out_idx = pid % num_out
    
    # Load Alpha (Current) - which is pot_out_ptr[out_idx]
    # Wait, we update in place? Or accumulate?
    # PyTorch implementation: alpha = alpha + delta.
    # So we need to load current alpha.
    alpha_ptr = pot_out_ptr + batch_idx * stride_pout_b + out_idx * stride_pout_d
    alpha_val = tl.load(alpha_ptr)
    
    marg_ptr = marginal_ptr + batch_idx * stride_marg_b + out_idx * stride_marg_d
    target_sum = tl.load(marg_ptr)
    
    # Accumulators
    sum_t_active = 0.0
    count_active = 0.0
    
    for off in range(0, num_in, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < num_in
        
        # Load Beta (pot_in)
        beta_val = tl.load(pot_in_ptr + batch_idx * stride_pin_b + cols * stride_pin_d, mask=mask, other=0.0)
        
        if is_row_update:
            c_ptr = C_ptr + batch_idx * stride_cb + out_idx * stride_cn + cols * stride_cm
        else:
            c_ptr = C_ptr + batch_idx * stride_cb + cols * stride_cn + out_idx * stride_cm
            
        c_val = tl.load(c_ptr, mask=mask, other=0.0)
        
        # T = alpha + beta - C
        T = alpha_val + beta_val - c_val
        
        # Active set
        # Note: If mask is false, we might process garbage which satisfies T>0?
        # Beta other=0, C other=0 -> T = alpha. If alpha > 0, false positive.
        # We must mask the T check.
        is_active = (T > 0) & mask
        
        # Accumulate Sum(T) where Active
        # tl.where(is_active, T, 0.0)
        curr_t = tl.where(is_active, T, 0.0)
        sum_t_active += tl.sum(curr_t, 0)
        
        # Accumulate Count
        curr_count = tl.where(is_active, 1.0, 0.0)
        count_active += tl.sum(curr_count, 0)
        
    # Scale by epsilon to get P
    current_sum_P = sum_t_active / epsilon
    
    # Gradient Ascent Step
    # grad = marginal - sum(P)
    # alpha += step * grad
    step_size = epsilon * 0.5
    
    grad_ascent = target_sum - current_sum_P
    delta = step_size * grad_ascent
    
    new_alpha = alpha_val + delta
    
    # Store updated alpha
    tl.store(alpha_ptr, new_alpha)


def l2_regularized_dual_ascent_triton(
    C: torch.Tensor,
    mu: torch.Tensor = None,
    nu: torch.Tensor = None,
    epsilon: float = 1.0,
    num_iters: int = 10
) -> torch.Tensor:
    
    if C.ndim == 2: C = C.unsqueeze(0)
    B, N, M = C.shape
    device = C.device
    
    if mu is None: mu = torch.ones(B, N, device=device) / N
    if nu is None: nu = torch.ones(B, M, device=device) / M
    
    alpha = torch.zeros(B, N, device=device, dtype=C.dtype)
    beta = torch.zeros(B, M, device=device, dtype=C.dtype)
    


    stride_cb, stride_cn, stride_cm = C.stride()
    
    for _ in range(num_iters):
        # Row Update: Updates alpha based on beta
        grid = (B * N,)
        _dual_ascent_update_kernel[grid](
            C, beta, alpha, mu,
            B, N, M,
            stride_cb, stride_cn, stride_cm,
            beta.stride(0), beta.stride(1),
            alpha.stride(0), alpha.stride(1),
            mu.stride(0), mu.stride(1),
            epsilon,
            True, # is_row_update
        )
        
        # Col Update: Updates beta based on alpha
        grid = (B * M,)
        _dual_ascent_update_kernel[grid](
            C, alpha, beta, nu,
            B, N, M,
            stride_cb, stride_cn, stride_cm,
            alpha.stride(0), alpha.stride(1),
            beta.stride(0), beta.stride(1),
            nu.stride(0), nu.stride(1),
            epsilon,
            False, # is_col_update
        )
        
    # Primal reconstruction
    T = alpha.unsqueeze(-1) + beta.unsqueeze(1) - C
    return torch.relu(T) / epsilon
