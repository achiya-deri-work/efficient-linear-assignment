import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _sinkhorn_update_kernel(
    C_ptr,           # (B, N, M)
    potential_src_ptr, # (B, N) - Input potential (g for row update, f for col update)
    potential_dst_ptr, # (B, M) - Output potential (f for row update, g for col update) - Actually shapes differ
    marginal_log_ptr,  # (B, OutDim) - log(mu) or log(nu)
    B, N, M,         # Dimensions
    stride_cb, stride_cn, stride_cm, # Strides for C
    stride_ps_b, stride_ps_d,        # Strides for input potential (Batch, Dim)
    stride_pd_b, stride_pd_d,        # Strides for output potential
    stride_m_b, stride_m_d,          # Strides for marginal
    epsilon,
    is_row_update: tl.constexpr, # If True: reduce over M (cols). Out: N.
    BLOCK_SIZE: tl.constexpr
):
    # If Row Update:
    # We want to compute f[b, n] = log(mu[b, n]) - logsumexp_j( (-C[b,n,j]/eps) + g[b, j] )
    # Input potential is g (size B, M). Output is f (size B, N).
    
    # If Col Update:
    # We want to compute g[b, m] = log(nu[b, m]) - logsumexp_i( (-C[b,i,m]/eps) + f[b, i] )
    # Input potential is f (size B, N). Output is g (size B, M).
    
    pid = tl.program_id(0)
    
    # Map pid to batch and dimension index
    # Total grid size: (B * OutDim)
    # E.g. Row update: Grid (B * N).
    
    if is_row_update:
        # Out Dimension is N. In Dimension is M.
        num_out = N
        num_in = M
    else:
        # Out Dimension is M. In Dimension is N.
        num_out = M
        num_in = N

    # Helper for indexing
    batch_idx = pid // num_out
    out_idx = pid % num_out
    
    # Pointer to Marginal
    marg_ptr = marginal_log_ptr + batch_idx * stride_m_b + out_idx * stride_m_d
    marg_val = tl.load(marg_ptr)
    
    # Init reduction
    # m_i and l_i for logsumexp
    # logsumexp(x) = m + log(sum(exp(x - m)))
    m_i = -float('inf')
    l_i = 0.0
    
    # Loop over inner dimension (blocks)
    for off in range(0, num_in, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < num_in
        
        # Load Input Potential (g for row update)
        # Shape (B, InDim).
        # Index: batch_id, cols
        pot_in_val = tl.load(potential_src_ptr + batch_idx * stride_ps_b + cols * stride_ps_d, mask=mask, other=-float('inf'))
        
        # Load C
        # If Row Update: C[b, out_idx, cols]
        # If Col Update: C[b, cols, out_idx]
        if is_row_update:
            # C [b, n, m] -> [b, out_idx, cols]
            c_ptr = C_ptr + batch_idx * stride_cb + out_idx * stride_cn + cols * stride_cm
        else:
            # C [b, n, m] -> [b, cols, out_idx]
            c_ptr = C_ptr + batch_idx * stride_cb + cols * stride_cn + out_idx * stride_cm
            
        c_val = tl.load(c_ptr, mask=mask, other=float('inf')) # other=inf so -C/eps is -inf
        
        # Compute term: -C/eps + pot
        # Note: c_val is Cost. We want -C/eps.
        term = (-c_val / epsilon) + pot_in_val
        
        # Online LSE
        m_next = tl.max(term, 0)
        # Update running max (m_i usually starts -inf, but tl.max needs tensor)
        # We need scalar max over block first?
        # No, standard LSE reduction pattern in Triton usually reduces block, then accumulates.
        # But here we assume BLOCK_SIZE covers the whole inner loop? 
        # Or we implement loop accumulation.
        # Simple loop accumulation:
        # m_curr = max(m_prev, block_max)
        # factor = exp(m_prev - m_curr)
        # sum = sum * factor + exp(x - m_curr)
        
        block_max = tl.max(term, 0)
        # If mask is used, masked items are -inf, so max is safe.
        
        # Update global max
        m_new = tl.maximum(m_i, block_max)
        
        # Update sum
        # exp terms
        # if m_new is -inf, everything is 0.
        term_exp = tl.exp(term - m_new)
        # Mask out-of-bounds
        term_exp = tl.where(mask, term_exp, 0.0)
        block_sum = tl.sum(term_exp, 0)
        
        l_new = l_i * tl.exp(m_i - m_new) + block_sum
        
        m_i = m_new
        l_i = l_new

    # Final Log Sum Exp
    # res = m_i + log(l_i)
    logsumexp_res = m_i + tl.log(l_i)
    
    # Result = log(marg) - logsumexp
    res_val = marg_val - logsumexp_res
    
    # Store Output Potential
    # Index: batch_idx, out_idx
    out_ptr = potential_dst_ptr + batch_idx * stride_pd_b + out_idx * stride_pd_d
    tl.store(out_ptr, res_val)

def log_stabilized_sinkhorn_triton(
    C: torch.Tensor,
    mu: torch.Tensor = None,
    nu: torch.Tensor = None,
    epsilon: float = 0.1,
    num_iters: int = 20
) -> torch.Tensor:
    
    if C.ndim == 2: C = C.unsqueeze(0)
    B, N, M = C.shape
    device = C.device
    
    if mu is None: mu = torch.ones(B, N, device=device) / N
    if nu is None: nu = torch.ones(B, M, device=device) / M
    
    log_mu = torch.log(mu + 1e-8)
    log_nu = torch.log(nu + 1e-8)
    
    # Potentials
    f = torch.zeros(B, N, device=device, dtype=C.dtype)
    g = torch.zeros(B, M, device=device, dtype=C.dtype)
    
    # Optimal Block Size: Next Power of 2 of Inner Dim
    # If N, M <= 1024, use 1024?
    # Or just fixed block size.
    BLOCK_SIZE = triton.next_power_of_2(max(N, M))
    if BLOCK_SIZE < 128: BLOCK_SIZE = 128
    # Triton limits block size? 128k threads? 
    # Usually max 1024 or 2048 threads per block.
    # If Dim > 2048, we need loop (implemented above).
    BLOCK_SIZE = min(BLOCK_SIZE, 1024) 
    
    # Strides
    stride_cb, stride_cn, stride_cm = C.stride()
    stride_ps_b, stride_ps_d = f.stride() # Initial shape assumption (B, N)
    stride_pd_b, stride_pd_d = g.stride()
    stride_m_b, stride_m_d = log_mu.stride()
    
    for _ in range(num_iters):
        # Row Update: f = log(mu) - LSE(M_eps + g)
        # Out: f (B, N). In: g (B, M). Inner: M (Cols).
        grid = (B * N,)
        _sinkhorn_update_kernel[grid](
            C, g, f, log_mu,
            B, N, M,
            stride_cb, stride_cn, stride_cm,
            g.stride(0), g.stride(1),
            f.stride(0), f.stride(1),
            log_mu.stride(0), log_mu.stride(1),
            epsilon,
            True, # is_row_update
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Col Update: g = log(nu) - LSE(M_eps + f)
        # Out: g (B, M). In: f (B, N). Inner: N (Rows).
        grid = (B * M,)
        # Note: Strides for C input? Kernel handles it via indexing.
        _sinkhorn_update_kernel[grid](
            C, f, g, log_nu,
            B, N, M,
            stride_cb, stride_cn, stride_cm,
            f.stride(0), f.stride(1),
            g.stride(0), g.stride(1),
            log_nu.stride(0), log_nu.stride(1),
            epsilon,
            False, # is_row_update
            BLOCK_SIZE=BLOCK_SIZE
        )
        
    # Final Plan construction?
    # PyTorch implementation returns exp(M_eps + f + g).
    # We can perform this in PyTorch or Triton.
    # For simplicity, PyTorch is fast enough for elementwise exp at end.
    # Or write a kernel.
    # Let's verify output.
    # M_eps = -C/eps
    # log_P = M_eps + f.unsqueeze(-1) + g.unsqueeze(1)
    
    return torch.exp((-C / epsilon) + f.unsqueeze(-1) + g.unsqueeze(1))
