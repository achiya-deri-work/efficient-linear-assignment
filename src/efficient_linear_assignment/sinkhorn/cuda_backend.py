import torch
from efficient_linear_assignment.auction.cpp_backend import efficient_linear_assignment_cpp

# Ensure extension is loaded
if not hasattr(efficient_linear_assignment_cpp, 'sinkhorn_cuda_forward'):
    raise ImportError("Sinkhorn CUDA kernel not found in efficient_linear_assignment_cpp. Rebuild extension.")

def log_stabilized_sinkhorn_cuda(
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
    
    # Kernel Forward
    # Returns [f, g]
    f, g = efficient_linear_assignment_cpp.sinkhorn_cuda_forward(
        C, log_mu, log_nu, epsilon, num_iters
    )
    
    # Primal Reconstruction
    # P = exp(-C/eps + f + g)
    return torch.exp((-C / epsilon) + f.unsqueeze(-1) + g.unsqueeze(1))
