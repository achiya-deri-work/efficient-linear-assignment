import torch
import efficient_linear_assignment.efficient_linear_assignment_cpp as efficient_linear_assignment_cpp

def l2_regularized_dual_ascent_cutlass(C, mu=None, nu=None, epsilon=1e-3, num_iters=10):
    B, N, M = C.shape
    if mu is None:
        mu = torch.ones(B, N, device=C.device, dtype=C.dtype) / N
    if nu is None:
        nu = torch.ones(B, M, device=C.device, dtype=C.dtype) / M
        
    # Call C++ binding
    # Expected signature: dual_ascent_cutlass_forward(C, mu, nu, epsilon, max_iter)
    alpha, beta = efficient_linear_assignment_cpp.dual_ascent_cutlass_forward(
        C, mu, nu, epsilon, num_iters
    )
    
    # P = ReLU( (alpha + beta - C)/eps )
    P = torch.relu( (alpha.unsqueeze(2) + beta.unsqueeze(1) - C) / epsilon )
    
    # Renormalize to ensure sum(P) = 1?
    # Usually Sinkhorn outputs satisfy constraints exactly if converged.
    # Dual Ascent might not sum to 1 exactly if not converged, but we return P as is.
    
    return P
