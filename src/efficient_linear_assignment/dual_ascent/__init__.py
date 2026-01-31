from .api import l2_regularized_dual_ascent

def l2_regularized_dual_ascent_dispatch(C, mu, nu, epsilon, num_iters, backend='cpu'):
    if backend == 'cpu':
        # Default/Torch implementation fallback if no specific CPU optimized version
        return l2_regularized_dual_ascent(C, mu, nu, epsilon, num_iters)
    elif backend == 'torch':
        return l2_regularized_dual_ascent(C, mu, nu, epsilon, num_iters)
    elif backend == 'triton':
        from .triton_backend import l2_regularized_dual_ascent_triton
        return l2_regularized_dual_ascent_triton(C, mu, nu, epsilon, num_iters)
    elif backend == 'cuda':
        from .cuda_backend import l2_regularized_dual_ascent_cuda
        return l2_regularized_dual_ascent_cuda(C, mu, nu, epsilon, num_iters)
    elif backend == 'cutlass':
        from .cutlass_backend import l2_regularized_dual_ascent_cutlass
        return l2_regularized_dual_ascent_cutlass(C, mu, nu, epsilon, num_iters)
    else:
        raise ValueError(f"Unknown backend: {backend}")
