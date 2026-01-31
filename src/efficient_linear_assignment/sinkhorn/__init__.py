from .api import log_stabilized_sinkhorn

def log_stabilized_sinkhorn_dispatch(C, mu, nu, epsilon, num_iters, backend='cpu'):
    if backend == 'cpu':
        from .cpu_backend import log_stabilized_sinkhorn_cpu
        return log_stabilized_sinkhorn_cpu(C, mu, nu, epsilon, num_iters)
    elif backend == 'torch':
        return log_stabilized_sinkhorn(C, mu, nu, epsilon, num_iters)
    elif backend == 'triton':
        from .triton_backend import log_stabilized_sinkhorn_triton
        return log_stabilized_sinkhorn_triton(C, mu, nu, epsilon, num_iters)
    elif backend == 'cuda':
        from .cuda_backend import log_stabilized_sinkhorn_cuda
        return log_stabilized_sinkhorn_cuda(C, mu, nu, epsilon, num_iters)
    elif backend == 'cutlass':
        from .cutlass_backend import log_stabilized_sinkhorn_cutlass
        return log_stabilized_sinkhorn_cutlass(C, mu, nu, epsilon, num_iters)
    else:
        raise ValueError(f"Unknown backend: {backend}")
