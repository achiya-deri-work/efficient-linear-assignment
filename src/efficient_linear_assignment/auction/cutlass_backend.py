import torch
import efficient_linear_assignment.efficient_linear_assignment_cpp as efficient_linear_assignment_cpp

def solve_auction_cutlass(C, epsilon=1.0, max_iter=500, match_thresh=None):
    """
    Solve linear assignment using CUTLASS-optimized Auction Algorithm.
    
    Args:
        C (torch.Tensor): Cost matrix [B, N, M]. 
                          Usually Auction minimizes cost, but our kernel logic assumes:
                          Value = -C - Price. So it effectively minimizes Cost C.
        epsilon (float): Min bid increment.
        max_iter (int): Maximum bidding iterations.
        
    Returns:
        torch.Tensor: Assignment indices [B, N] or similar.
    """
    if C.is_cuda:
        # C++ extension handles BFloat16/Half/Float dispatch
        # Returned: assignments, prices
        # assignments: [B, N] of int32.
        assignments, prices = efficient_linear_assignment_cpp.auction_cutlass_forward(
            C, epsilon, max_iter
        )
        
        # Convert assignments to standard P matrix [B, N, M] if needed?
        # Or return indices?
        # Standard API usually returns 'P' (permutation matrix).
        # We will create P from indices.
        
        # Returned: assignments, prices
        # assignments: [B, N] of int32.
        
        # We must return indices [B, N] to be compatible with api.py
        # Also ensure it is LongTensor
        return assignments.long()
        
    else:
        raise NotImplementedError("CUTLASS backend only supports CUDA tensors")
