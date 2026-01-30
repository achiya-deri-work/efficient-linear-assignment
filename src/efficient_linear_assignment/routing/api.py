from .torch_backend import max_score_routing as max_score_routing_torch

BACKENDS = {
    'torch': max_score_routing_torch,
}

try:
    from .triton_backend import max_score_routing_triton
    BACKENDS['triton'] = max_score_routing_triton
except ImportError:
    pass
    
try:
    from .cuda_backend import max_score_routing_cuda
    BACKENDS['cuda'] = max_score_routing_cuda
except ImportError:
    pass

def max_score_routing(logits, capacity_factor=1.0, epsilon=0.1, num_iters=15, backend=None):
    if backend is None:
        backend = 'torch'
        
    if backend not in BACKENDS:
         raise ValueError(f"Backend '{backend}' not available. Choices: {list(BACKENDS.keys())}")
         
    return BACKENDS[backend](logits, capacity_factor, epsilon, num_iters)
