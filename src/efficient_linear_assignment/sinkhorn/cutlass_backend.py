import torch
import efficient_linear_assignment.efficient_linear_assignment_cpp as efficient_linear_assignment_cpp

def log_stabilized_sinkhorn_cutlass(C, mu=None, nu=None, epsilon=1e-3, num_iters=20):
    B, N, M = C.shape
    if mu is None:
        mu = torch.ones(B, N, device=C.device, dtype=C.dtype) / N
    if nu is None:
        nu = torch.ones(B, M, device=C.device, dtype=C.dtype) / M
        
    log_mu = torch.log(mu)
    log_nu = torch.log(nu)
    
    # Call C++ binding
    # Expected signature: sinkhorn_cutlass_forward(C, log_mu, log_nu, epsilon, max_iter)
    f, g = efficient_linear_assignment_cpp.sinkhorn_cutlass_forward(
        C, log_mu, log_nu, epsilon, num_iters
    )
    
    # Compute P = exp( (f + g - C)/eps )
    # P = torch.exp( (f.unsqueeze(2) + g.unsqueeze(1) - C) / epsilon )
    # But usually we return log_P or P.
    # The interface of log_stabilized_sinkhorn usually returns P.
    
    P = torch.exp(f.unsqueeze(2) + g.unsqueeze(1) - (C / epsilon))
    return P
