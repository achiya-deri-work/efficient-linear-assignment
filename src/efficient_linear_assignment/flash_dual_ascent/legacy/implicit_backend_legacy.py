import torch

def l2_regularized_dual_ascent_implicit(Q, K, mu=None, nu=None, epsilon=1.0, num_iters=10):
    """
    Solves Dual Ascent using Implicit Costs (Q @ K.T).
    Args:
        Q: [B, N, D]
        K: [B, M, D]
        ...
    """
    try:
        import efficient_linear_assignment.efficient_linear_assignment_cpp as cpp
    except ImportError:
        raise ImportError("C++ extension not built.")
        
    if Q.dim() != 3 or K.dim() != 3:
        raise ValueError("Q and K must be 3D tensors (Batch, N, D)")
    
    Q = Q.contiguous()
    K = K.contiguous()
    
    B, N, D = Q.shape
    M = K.shape[1]
    
    if mu is None: mu = torch.ones(B, N, device=Q.device, dtype=Q.dtype) / N
    if nu is None: nu = torch.ones(B, M, device=Q.device, dtype=Q.dtype) / M
    
    mu = mu.contiguous()
    nu = nu.contiguous()
    
    # Call C++
    alpha, beta = cpp.dual_ascent_implicit_forward(Q, K, mu, nu, epsilon, num_iters)
    
    return alpha, beta

def l2_regularized_dual_ascent_implicit_v2(Q, K, mu=None, nu=None, epsilon=1.0, num_iters=10):
    try:
         import efficient_linear_assignment.efficient_linear_assignment_cpp as cpp
    except ImportError:
         raise ImportError("C++ extension not built.")

    if Q.dim() != 3 or K.dim() != 3:
        raise ValueError("Q and K must be 3D tensors (Batch, N, D)")
    
    Q = Q.contiguous()
    K = K.contiguous()
    
    B, N, D = Q.shape
    M = K.shape[1]
    
    if mu is None: mu = torch.ones(B, N, device=Q.device, dtype=Q.dtype) / N
    if nu is None: nu = torch.ones(B, M, device=Q.device, dtype=Q.dtype) / M

    mu = mu.contiguous()
    nu = nu.contiguous()

    # Call C++ v2
    return cpp.dual_ascent_implicit_v2_forward(
        Q, K, mu, nu, epsilon, num_iters
    )
