from .torch_backend import log_stabilized_sinkhorn as log_stabilized_sinkhorn_torch

BACKENDS = {
    'torch': log_stabilized_sinkhorn_torch,
}

# Try importing other backends
try:
    from .triton_backend import log_stabilized_sinkhorn_triton
    BACKENDS['triton'] = log_stabilized_sinkhorn_triton
except ImportError:
    pass

try:
    from .cuda_backend import log_stabilized_sinkhorn_cuda
    BACKENDS['cuda'] = log_stabilized_sinkhorn_cuda
except ImportError:
    pass

def log_stabilized_sinkhorn(C, mu=None, nu=None, epsilon=0.1, num_iters=20, backend=None):
    if backend is None:
        backend = 'torch' # Default
    
    if backend not in BACKENDS:
         raise ValueError(f"Backend '{backend}' not available. Choices: {list(BACKENDS.keys())}")
         
    return BACKENDS[backend](C, mu, nu, epsilon, num_iters)
