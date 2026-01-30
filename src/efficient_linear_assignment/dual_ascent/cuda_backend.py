import torch
from efficient_linear_assignment.auction.cpp_backend import efficient_linear_assignment_cpp

if not hasattr(efficient_linear_assignment_cpp, 'dual_ascent_cuda_forward'):
    raise ImportError("Dual Ascent CUDA kernel not found in extension.")

def l2_regularized_dual_ascent_cuda(
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
    
    # Returns [alpha, beta]
    alpha, beta = efficient_linear_assignment_cpp.dual_ascent_cuda_forward(
        C, mu, nu, epsilon, num_iters
    )
    
    # Primal Reconstruction
    # T = alpha + beta - C
    # P = ReLU(T) / eps
    T = alpha.unsqueeze(-1) + beta.unsqueeze(1) - C
    return torch.relu(T) / epsilon
