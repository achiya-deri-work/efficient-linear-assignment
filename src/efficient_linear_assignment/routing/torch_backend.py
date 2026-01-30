import torch
from efficient_linear_assignment.dual_ascent import l2_regularized_dual_ascent

def max_score_routing(
    logits: torch.Tensor,
    capacity_factor: float = 1.0,
    epsilon: float = 0.1,
    num_iters: int = 15
) -> torch.Tensor:
    """
    MaxScore Routing via Capacity-Constrained L2-OT.
    
    Treats routing as a global assignment problem to maximize total score
    subject to expert capacity constraints. Uses L2 regularization to 
    maintain sparsity (SoftTopk behavior).
    
    Args:
        logits: Affinities/Scores (Batch, Tokens, Experts).
        capacity_factor: Multiplier for expert buffer size.
                         (1.0 = balanced load, >1.0 = slack).
        epsilon: Temperature/Regularization.
        num_iters: Solver iterations.
        
    Returns:
        P: Sparse routing weights (Batch, Tokens, Experts).
    """
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
        
    B, Tokens, Experts = logits.shape
    device = logits.device
    dtype = logits.dtype
    
    # 1. Constraints
    # Source: Every token processes mass 1.0
    mu = torch.ones(B, Tokens, device=device, dtype=dtype)
    
    # Target: Experts have limited capacity
    # Global capacity = (Tokens / Experts) * Factor
    cap = (Tokens / Experts) * capacity_factor
    nu = torch.full((B, Experts), cap, device=device, dtype=dtype)
    
    # 2. Cost definition
    # MaxScore(logits) == MinCost(-logits)
    C = -logits
    
    # 3. Solve via L2-Regularized Dual Ascent
    # This produces a sparse plan where low-scoring token-expert pairs 
    # are exactly zero, but gradients flow through the active set.
    P = l2_regularized_dual_ascent(C, mu, nu, epsilon, num_iters)
    
    # 4. Normalize
    # L2-OT satisfies constraints approximately. Normalize to ensure
    # strictly valid routing weights.
    row_sums = P.sum(dim=2, keepdim=True)
    mask_zero = row_sums < 1e-6
    if mask_zero.any():
        B, T, E = P.shape
        uniform = torch.ones_like(P) / E
        P = torch.where(mask_zero, uniform, P)
        row_sums = P.sum(dim=2, keepdim=True)
         
    P = P / (row_sums + 1e-8)
    
    return P
