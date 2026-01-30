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
        # --- Row Update (Newton Step) ---
        # Current plan P = ReLU(alpha + beta - C) / epsilon
        T = alpha + beta - C
        active_mask = (T > 0).float() # Keeps gradient flow? No, mask is discrete.
        # But in backward pass, gradients flow through alpha/beta that made T > 0.
        # This is essentially ReLU gradient.
        
        # Calculate current mass and gradient (count of active elements)
        current_sum = (T * active_mask).sum(dim=2, keepdim=True) / epsilon
        active_count = active_mask.sum(dim=2, keepdim=True)
        
        # Newton Update: alpha += (Target - Current) / Gradient
        # Clamp gradient to avoid division by zero for inactive rows
        grad = active_count.clamp(min=1e-6) / epsilon
        delta = (mu.unsqueeze(-1) - current_sum) / grad
        alpha = alpha + delta
        
        # --- Column Update (Newton Step) ---
        # Recompute T with updated alpha
        T = alpha + beta - C
        active_mask = (T > 0).float()
        
        current_sum = (T * active_mask).sum(dim=1, keepdim=True) / epsilon
        active_count = active_mask.sum(dim=1, keepdim=True)
        
        grad = active_count.clamp(min=1e-6) / epsilon
        delta = (nu.unsqueeze(1) - current_sum) / grad
        beta = beta + delta
        
    # Final projection
    # Force output to be max(0, X)
    return torch.relu(alpha + beta - C) / epsilon
