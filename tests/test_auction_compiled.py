
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from efficient_linear_assignment.compiled import auction_compiled
import pytest
import time

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_auction_compiled():
    """
    Verifies the compiled Auction implementation against Scipy (correctness)
    and checks output validity.
    """
    # Reduce N for faster verification
    B, N = 2, 16
    device = 'cuda'
    
    # Generate Cost Matrix
    torch.manual_seed(42)
    C = torch.rand(B, N, N, device=device)
    
    # Run Compiled Auction
    # N=16. iter ~ 16*100=1600. Using 5000.
    assignments = auction_compiled(C, epsilon=1e-2, max_iter=5000)
    
    # Verify Validity (Permutation)
    if isinstance(assignments, tuple):
        assignments = assignments[0] # Assume first element is assignment
    
    assert assignments.shape == (B, N)
    
    failures = []
    
    for b in range(B):
        assign = assignments[b].cpu().numpy()
        cost_mat = C[b].cpu().numpy()
        
        # Check Uniqueness / Validity
        unassigned_count = np.sum(assign == -1)
        if unassigned_count > 0:
            failures.append(f"Batch {b}: found {unassigned_count} unassigned agents")
            continue
            
        unique_targets = len(np.unique(assign))
        if unique_targets != N:
            failures.append(f"Batch {b}: NOT a permutation! Unique targets: {unique_targets}")
            continue
        
        # Check Optimality vs Scipy
        try:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            scipy_cost = cost_mat[row_ind, col_ind].sum()
            
            my_cost = cost_mat[np.arange(N), assign].sum()
            
            diff = abs(my_cost - scipy_cost)
            
            # Auction is approximate (within N * eps).
            # Eps = 1e-2. N = 64. Error bound ~ 0.64.
            bound = N * 2e-2
            
            if diff > bound:
                failures.append(f"Batch {b}: Cost diff {diff:.6f} > bound {bound:.6f} (Scipy={scipy_cost:.4f}, My={my_cost:.4f})")
                
        except Exception as e:
            failures.append(f"Batch {b}: Scipy Comparison Failed: {e}")
            
    assert len(failures) == 0, "\n".join(failures)

if __name__ == "__main__":
    test_auction_compiled()
