from .torch_backend import l2_regularized_dual_ascent as l2_regularized_dual_ascent_torch

BACKENDS = {
    'torch': l2_regularized_dual_ascent_torch,
}

try:
    from .triton_backend import l2_regularized_dual_ascent_triton
    BACKENDS['triton'] = l2_regularized_dual_ascent_triton
except ImportError:
    pass

try:
    from .cuda_backend import l2_regularized_dual_ascent_cuda
    BACKENDS['cuda'] = l2_regularized_dual_ascent_cuda
except ImportError:
    pass

try:
    from .cutlass_backend import l2_regularized_dual_ascent_cutlass
    BACKENDS['cutlass'] = l2_regularized_dual_ascent_cutlass
except ImportError:
    pass

def l2_regularized_dual_ascent(C, mu=None, nu=None, epsilon=1.0, num_iters=10, backend=None):
    if backend is None:
        backend = 'torch'
        
    if backend not in BACKENDS:
         raise ValueError(f"Backend '{backend}' not available. Choices: {list(BACKENDS.keys())}")
         
    return BACKENDS[backend](C, mu, nu, epsilon, num_iters)
