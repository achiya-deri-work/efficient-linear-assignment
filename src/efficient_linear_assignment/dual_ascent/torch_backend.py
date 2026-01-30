import torch

def l2_regularized_dual_ascent(
    C: torch.Tensor,
    mu: torch.Tensor = None,
    nu: torch.Tensor = None,
    epsilon: float = 1.0,
    num_iters: int = 10
) -> torch.Tensor:
    """
    Computes Sparse Optimal Transport via L2-Regularized Dual Ascent.
    Formula: P = ReLU(alpha + beta - C) / epsilon
    
    Args:
        C: Cost matrix (Batch, N, M).
        mu, nu: Source/Target marginals.
        epsilon: Regularization strength. Controls sparsity.
        num_iters: Number of Newton coordinate updates.
        
    Returns:
        P: Sparse transport plan (Batch, N, M).
    """
    if C.ndim == 2:
        C = C.unsqueeze(0)
    
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype
    
    if mu is None: 
        mu = torch.ones(B, N, device=device, dtype=dtype) / N
    if nu is None: 
        nu = torch.ones(B, M, device=device, dtype=dtype) / M

    # Initialize dual potentials
    alpha = torch.zeros(B, N, 1, device=device, dtype=dtype)
    beta = torch.zeros(B, 1, M, device=device, dtype=dtype)
    
    for _ in range(num_iters):
        # Gradient Ascent Logic (Stable)
        step_size = epsilon * 0.5
        
        # --- Row Update ---
        T = alpha + beta - C
        P = torch.relu(T) / epsilon
        current_sum = P.sum(dim=2, keepdim=True)
        alpha = alpha + step_size * (mu.unsqueeze(-1) - current_sum)
        
        # --- Column Update ---
        T = alpha + beta - C
        P = torch.relu(T) / epsilon
        current_sum = P.sum(dim=1, keepdim=True)
        beta = beta + step_size * (nu.unsqueeze(1) - current_sum)
        
    # Final projection
    # Force output to be max(0, X)
    return torch.relu(alpha + beta - C) / epsilon
