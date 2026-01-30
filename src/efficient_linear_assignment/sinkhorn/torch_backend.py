import torch
import torch.nn.functional as F

def log_stabilized_sinkhorn(
    C: torch.Tensor,
    mu: torch.Tensor = None,
    nu: torch.Tensor = None,
    epsilon: float = 0.1,
    num_iters: int = 20
) -> torch.Tensor:
    """
    Computes the Entropic Optimal Transport plan using Log-Stabilized Sinkhorn iterations.
    
    Args:
        C: Cost matrix of shape (Batch, N, M).
        mu: Source marginals (Batch, N). Defaults to uniform (1/N).
        nu: Target marginals (Batch, M). Defaults to uniform (1/M).
        epsilon: Entropic regularization strength. Higher = smoother plan.
        num_iters: Number of unrolled iterations.

    Returns:
        P: The dense optimal transport plan (Batch, N, M).
    """
    # Ensure Batch Dim
    if C.ndim == 2:
        C = C.unsqueeze(0)
    
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype
    
    # Initialize marginals to uniform if not provided
    if mu is None: 
        mu = torch.ones(B, N, device=device, dtype=dtype) / N
    if nu is None: 
        nu = torch.ones(B, M, device=device, dtype=dtype) / M

    # Work in log-domain for numerical stability
    # Add epsilon to log to prevent -inf
    log_mu = torch.log(mu + 1e-8)
    log_nu = torch.log(nu + 1e-8)
    
    # Initialize dual potentials (log(u), log(v))
    # We keep them dimensionless (implicitly scaled by epsilon in the update)
    f = torch.zeros(B, N, 1, device=device, dtype=dtype)
    g = torch.zeros(B, 1, M, device=device, dtype=dtype)
    
    # Pre-compute scaled cost for the LSE kernel
    M_eps = -C / epsilon
    
    for _ in range(num_iters):
        # Row Update (f): Match row marginals
        # f = log(mu) - logsumexp( M_eps + g )
        # Note: M_eps + g broadcasts to (B, N, M)
        f = log_mu.unsqueeze(-1) - torch.logsumexp(M_eps + g, dim=2, keepdim=True)
        
        # Column Update (g): Match column marginals
        # g = log(nu) - logsumexp( M_eps + f )
        g = log_nu.unsqueeze(1) - torch.logsumexp(M_eps + f, dim=1, keepdim=True)
        
    # Reconstruct primal plan P = exp(f + g - C/eps)
    log_P = M_eps + f + g
    return torch.exp(log_P)
