import torch
from typing import Optional, Tuple
from .torch_backend import AuctionTorch

# Dictionary to hold backend implementations. 
# Populated by imports or explicit registration.
BACKENDS = {
    'torch': AuctionTorch,
}

# Try importing other backends gracefully
try:
    from .triton_backend import AuctionTriton
    BACKENDS['triton'] = AuctionTriton
except ImportError:
    pass

try:
    from .cpp_backend import AuctionCPPCCUDA
    BACKENDS['cpp'] = AuctionCPPCCUDA
except ImportError:
    pass

try:
    from .cutile_backend import AuctionCuTile
    BACKENDS['cutile'] = AuctionCuTile
except ImportError:
    pass

try:
    from .cutlass_backend import solve_auction_cutlass
    # Wrapper helper to match interface
    class AuctionCutlass:
        def __init__(self, epsilon, max_iter):
            self.epsilon = epsilon
            self.max_iter = max_iter
        def solve(self, cost_matrix):
            # returns indices (B, N)
            indices = solve_auction_cutlass(cost_matrix, self.epsilon, self.max_iter)
            return indices, None # No prices for now
    BACKENDS['cutlass'] = AuctionCutlass
    BACKENDS['cutlass'] = AuctionCutlass
except ImportError:
    pass





def linear_assignment(cost_matrix: torch.Tensor, epsilon: float = 1e-2, max_iter: int = 1000, backend: str = 'torch', return_indices: bool = True):
    """
    Solves LAP using the Auction Algorithm.
    Args:
        cost_matrix: (B, N, M)
        return_indices: If True, returns (B, N) LongTensor.
    """
    # Ensure Batch Dim
    is_batched = cost_matrix.ndim == 3
    if not is_batched:
        cost_matrix = cost_matrix.unsqueeze(0)
    
    # Validate dimensions (Multiples of 8 check)
    _, N, M = cost_matrix.shape
    if N % 8 != 0 or M % 8 != 0:
        raise ValueError(f"Input dimensions must be multiples of 8. Got shape {cost_matrix.shape}. Please pad your input.")

    # Select Backend
    if backend not in BACKENDS:
        raise ValueError(f"Backend '{backend}' not available. Options: {list(BACKENDS.keys())}")
    
    backend_cls = BACKENDS[backend]
    solver = backend_cls(epsilon=epsilon, max_iter=max_iter)

    # Solve
    # returns indices (B, N) usually, or (indices, prices) depending on backend.
    # We unify to return indices.
    result = solver.solve(cost_matrix)
    
    if isinstance(result, tuple):
        assignment_indices = result[0]
    else:
        assignment_indices = result
        
    if not return_indices:
        # Convert to matrix if requested? (Legacy support)
        # We can reconstruct P from indices
        B, N, M = cost_matrix.shape
        assignment_matrix = torch.zeros_like(cost_matrix)
        batch_idx = torch.arange(B, device=cost_matrix.device).unsqueeze(1)
        row_idx = torch.arange(N, device=cost_matrix.device).unsqueeze(0)
        
        valid_mask = (assignment_indices >= 0)
        # clamp to avoid -1 index error
        safe_indices = assignment_indices.clamp(0)
        
        assignment_matrix[batch_idx, row_idx, safe_indices] = valid_mask.to(cost_matrix.dtype)
        
        if not is_batched:
            assignment_matrix = assignment_matrix.squeeze(0)
        return assignment_matrix

    if not is_batched:
        assignment_indices = assignment_indices.squeeze(0)
        
    return assignment_indices
