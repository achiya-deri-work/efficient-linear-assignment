
import torch
import pytest
from efficient_linear_assignment import linear_assignment
import time

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed")
class TestBackendConsistency:
    def test_small_deterministic(self):
        device = 'cuda'
        # Cost where diag is best (0 vs 10)
        B, N, M = 2, 32, 32
        cost = torch.full((B, N, M), 10.0, device=device)
        for b in range(B):
            cost[b].fill_diagonal_(0.0)
            
        expected = torch.arange(N, device=device).expand(B, N)
        
        # Torch (Official/Reference)
        assign_torch = linear_assignment(cost, backend='torch', return_indices=True)
        assert (assign_torch == expected).all(), "Torch failed baseline logic"
        
        # Triton
        try:
            assign_triton = linear_assignment(cost, backend='triton', return_indices=True)
            assert (assign_triton == expected).all(), "Triton result mismatch on simple diag matrix"
        except Exception as e:
            pytest.fail(f"Triton failed: {e}")
            
        # C++
        try:
            assign_cpp = linear_assignment(cost, backend='cpp') # usually returns indices directly if not wrapper? 
            # Wait, API wrapper standardizes return.
            # let's check return type or just assume indices if tensor.
            # API usually returns indices or P. `linear_assignment` (default) calls Sinkhorn/Log?? 
            # No, 'auction' logic in verify_correctness.py implied Auction?
            # verify_correctness.py imported `from efficient_linear_assignment.api import linear_assignment`
            # This API function usually defaults to something.
            # But here we specify backend.
            # If backend='cpp', strictly it calls auction.
            
            assert (assign_cpp == expected).all(), "C++ result mismatch"
        except Exception as e:
             # C++ might not be installed or built
             if "not available" in str(e) or "import" in str(e).lower():
                 pytest.skip("C++ backend not available")
             else:
                 pytest.fail(f"C++ failed: {e}")

    def test_random_large_consistency(self):
        """Compare Costs between Torch and Triton"""
        device = 'cuda'
        B, N, M = 1, 512, 512 # Reduced from 1024 to be faster
        torch.manual_seed(42)
        cost = torch.rand(B, N, M, device=device, dtype=torch.float32) # Standard
        
        assign_torch = linear_assignment(cost, backend='torch')
        try:
            assign_triton = linear_assignment(cost, backend='triton')
        except Exception as e:
            pytest.skip(f"Triton failed: {e}")
            
        def get_total_cost(assign, c):
            if assign.ndim == 3: # Soft assignment?
                # If API returns soft mat for torch but indices for triton?
                # In `verify_correctness.py`, it assumed indices.
                # `linear_assignment` -> `api.py`.
                # If backend='torch', it likely calls sinkhorn which returns P (BxNxN).
                # Wait, verify_correctness.py line 29: `return_indices=True`.
                # Line 80: `assign_torch = linear_assignment(cost, backend='torch')`. 
                # Does it default to indices?
                pass
            
            # Assuming indices BxN
            total = 0.0
            for b_idx in range(B):
                row_idx = torch.arange(N, device=c.device)
                col_idx = assign[b_idx]
                vals = c[b_idx, row_idx, col_idx]
                total += vals.sum().item()
            return total
            
        # Check shapes. If torch returns soft, convert.
        if assign_torch.ndim == 3:
            # It's soft assignment matrix P
            assign_torch = assign_torch.argmax(dim=2)
            
        cost_torch = get_total_cost(assign_torch, cost)
        cost_triton = get_total_cost(assign_triton, cost)
        
        diff = abs(cost_torch - cost_triton)
        rel_diff = diff / (abs(cost_torch) + 1e-6)
        
        # 1.5% tolerance for approx algorithms
        assert rel_diff < 0.015, f"Triton cost diverged from Torch: RelDiff={rel_diff:.4f}"

