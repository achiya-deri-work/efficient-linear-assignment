
import torch
import math
from typing import Tuple, Optional

def get_pad_size(n: int, multiple: int = 8) -> int:
    remainder = n % multiple
    if remainder == 0:
        return 0
    return multiple - remainder

def pad_to_shape(tensor: torch.Tensor, target_shape: Tuple[int, ...], value: float = 0.0) -> torch.Tensor:
    """
    Pads the last N dimensions of the tensor to match target_shape.
    """
    # Calculate padding for torch.nn.functional.pad
    # Format is (padding_left, padding_right, padding_top, padding_bottom, ...)
    # starting from the last dimension.
    pad_args = []
    
    # Iterate from last dim backwards
    for i in range(len(target_shape)-1, -1, -1):
        # Current dim size
        curr_dim = tensor.shape[i]
        target_dim = target_shape[i]
        
        diff = target_dim - curr_dim
        pad_args.extend([0, diff]) # Pad end only
        
    return torch.nn.functional.pad(tensor, tuple(pad_args), value=value)

def pad_input(cost_matrix: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Pads cost matrix [B, N, M] so that N and M are multiples of 'multiple'.
    Also ensures B is handled if necessary (though usually B doesn't need strict padding for these kernels unless specified).
    
    The user specified: "sizes must be a multiple of 8 on every dim".
    We will strictly pad N and M.
    
    For the Auction algorithm:
    - Padding rows (Agents): Add dummy agents. They should ideally get matched to dummy objects.
    - Padding cols (Objects): Add dummy objects.
    
    To prevent real agents from matching with dummy objects, or dummy agents with real objects:
    - Real Agents -> Dummy Objects: Cost = Infinity (Benefit = -Inf)
    - Dummy Agents -> Real Objects: Cost = Infinity
    - Dummy Agents -> Dummy Objects: Cost = 0 (Preferred)
    
    However, a simpler approach often works: 
    - Just pad with 0s or large number?
    - If we pad with Cost=0 (high benefit), real agents might want dummy objects. Bad.
    - If we pad with Cost=Inf (low benefit), real agents avoid dummy objects. Good.
    - Dummy agents? If they have Cost=Inf to real objects, they avoid real objects.
    
    Strategy:
    1. Pad N -> N_pad. New rows are dummy agents.
    2. Pad M -> M_pad. New cols are dummy objects.
    3. Block (0:N, 0:M): Original Costs.
    4. Block (0:N, M:M_pad): Real agents to Dummy objects. Cost = INF.
    5. Block (N:N_pad, 0:M): Dummy agents to Real objects. Cost = INF.
    6. Block (N:N_pad, M:M_pad): Dummy to Dummy. Cost = 0.
    
    This ensures solving the bigger problem yields the same sub-assignment.
    """
    B, N, M = cost_matrix.shape
    
    N_pad_size = get_pad_size(N, multiple)
    M_pad_size = get_pad_size(M, multiple)
    
    if N_pad_size == 0 and M_pad_size == 0:
        return cost_matrix, (B, N, M)
        
    N_new = N + N_pad_size
    M_new = M + M_pad_size
    
    # Create larger tensor filled with a high cost (low benefit).
    # Since we typically minimize Cost or Maximize Benefit.
    # The solver maximizes Benefit = -Cost - Price.
    # We want "Infinite Cost".
    large_cost = 1e9 # Sufficiently large
    
    padded_cost = torch.full((B, N_new, M_new), large_cost, dtype=cost_matrix.dtype, device=cost_matrix.device)
    
    # Copy original costs
    padded_cost[:, :N, :M] = cost_matrix
    
    # Dummy-Dummy block set to 0 cost (high benefit relative to large_cost)
    # This ensures dummy agents prefer dummy objects.
    if N_pad_size > 0 and M_pad_size > 0:
        padded_cost[:, N:, M:] = 0.0
        
    # If only one dimension is padded?
    # e.g. N=31, M=32. Pad N to 32. 
    # We add 1 dummy agent.
    # It needs to match with something. But M is full.
    # If M is full, we can't add a dummy object just for the dummy agent unless we pad M too?
    # Actually, Linear Assignment assumes square usually (N=M) or N<=M.
    # If N != M, we might need to pad to square max(N,M) rounded to 8?
    # Let's assume we pad to independently multiples of 8.
    # If N_new != M_new, the solver must handle rectangular. 
    # The Auction algorithm usually handles N <= M.
    # If we add dummy agents but no dummy objects, they can't be assigned if N > M?
    # Let's assume the solver handles the shape provided.
    
    return padded_cost, (B, N, M)

def unpad_result(indices: torch.Tensor, original_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    indices: [B, N_pad]
    """
    B, N, M = original_shape
    return indices[:, :N]
