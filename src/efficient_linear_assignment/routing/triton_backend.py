from efficient_linear_assignment.dual_ascent.triton_backend import l2_regularized_dual_ascent_triton
from .torch_backend import max_score_routing as max_score_routing_torch # Reuse normalization logic?
# Or reimplement wrapper. Wrappers are light.

import torch

def max_score_routing_triton(
    logits: torch.Tensor,
    capacity_factor: float = 1.0,
    epsilon: float = 0.1,
    num_iters: int = 15
) -> torch.Tensor:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
        
    B, Tokens, Experts = logits.shape
    device = logits.device
    dtype = logits.dtype
    
    mu = torch.ones(B, Tokens, device=device, dtype=dtype)
    cap = (Tokens / Experts) * capacity_factor
    nu = torch.full((B, Experts), cap, device=device, dtype=dtype)
    
    C = -logits
    
    # Call Triton Dual Ascent
    P = l2_regularized_dual_ascent_triton(C, mu, nu, epsilon, num_iters)
    
    # Normalize (Reuse torch logic for vectorization)
    row_sums = P.sum(dim=2, keepdim=True)
    mask_zero = row_sums < 1e-6
    if mask_zero.any():
        E = Experts
        uniform = torch.ones_like(P) / E
        P = torch.where(mask_zero, uniform, P)
        row_sums = P.sum(dim=2, keepdim=True)
         
    P = P / (row_sums + 1e-8)
    
    return P
