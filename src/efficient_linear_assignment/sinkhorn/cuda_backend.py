import torch
# Defer import to avoid circular dependency and initialization order issues
efficient_linear_assignment_cpp = None

def log_stabilized_sinkhorn_cuda(
    C: torch.Tensor,
    mu: torch.Tensor = None,
    nu: torch.Tensor = None,
    epsilon: float = 0.1,
    num_iters: int = 20
) -> torch.Tensor:
    
    global efficient_linear_assignment_cpp
    if efficient_linear_assignment_cpp is None:
        try:
             import efficient_linear_assignment.efficient_linear_assignment_cpp as efficient_linear_assignment_cpp
        except ImportError:
             raise ImportError("Failed to import efficient_linear_assignment_cpp extension")

    if C.ndim == 2: C = C.unsqueeze(0)
    B, N, M = C.shape
    device = C.device
    
    if mu is None: mu = torch.ones(B, N, device=device) / N
    if nu is None: nu = torch.ones(B, M, device=device) / M
    
    log_mu = torch.log(mu + 1e-8)
    log_nu = torch.log(nu + 1e-8)
    
    # Kernel Forward
    # Returns [f, g]
    f, g = efficient_linear_assignment_cpp.sinkhorn_cuda_forward(
        C, log_mu, log_nu, epsilon, num_iters
    )
    
    # Primal Reconstruction
    # P = exp(-C/eps + f + g)
    return torch.exp((-C / epsilon) + f.unsqueeze(-1) + g.unsqueeze(1))
