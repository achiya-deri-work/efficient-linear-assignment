import unittest
import torch
import torch.nn.functional as F
import sys
import os

# sys.path.append("src") # Removed to use installed package


from efficient_linear_assignment.sinkhorn import log_stabilized_sinkhorn
from efficient_linear_assignment.dual_ascent import l2_regularized_dual_ascent
from efficient_linear_assignment.routing import max_score_routing

class TestAlgos(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _test_sinkhorn(self, backend):
        print(f"Testing Sinkhorn [{backend}]")
        B, N = 2, 5
        torch.manual_seed(42)
        C = torch.randn(B, N, N, device=self.device)
        mu = torch.ones(B, N, device=self.device) / N
        nu = torch.ones(B, N, device=self.device) / N
        
        try:
            P = log_stabilized_sinkhorn(C, mu, nu, epsilon=0.1, num_iters=100, backend=backend)
        except Exception as e:
            if "not available" in str(e) or "not found" in str(e):
                print(f"Skipping {backend}: {e}")
                return
            raise e
            
        row_sums = P.sum(dim=2)
        col_sums = P.sum(dim=1)
        
        # Check constraints
        diff_row = (row_sums - mu).abs().max().item()
        diff_col = (col_sums - nu).abs().max().item()
        
        print(f"  [{backend}] Max Diff Row: {diff_row:.6f}")
        print(f"  [{backend}] Max Diff Col: {diff_col:.6f}")
        
        self.assertTrue(diff_row < 1e-2, f"{backend} Row constraint failed")
        self.assertTrue(diff_col < 1e-2, f"{backend} Col constraint failed")

    def test_sinkhorn_all(self):
        for backend in ['torch', 'triton', 'cuda']:
            self._test_sinkhorn(backend)

    def _test_dual_ascent(self, backend):
        print(f"Testing Dual Ascent [{backend}]")
        B, N = 2, 5
        torch.manual_seed(42)
        C = torch.rand(B, N, N, device=self.device) # Costs > 0 usually
        mu = torch.ones(B, N, device=self.device) / N
        nu = torch.ones(B, N, device=self.device) / N
        
        # Dual Ascent might not exactly satisfy constraints with small epsilon/iter
        # But should be close or at least run.
        try:
            P = l2_regularized_dual_ascent(C, mu, nu, epsilon=0.1, num_iters=50, backend=backend)
        except Exception as e:
            if "not available" in str(e):
                print(f"Skipping {backend}: {e}")
                return
            raise e

        # Basic check: shape
        self.assertEqual(P.shape, (B, N, N))
        # Non-negative
        self.assertTrue((P >= -1e-6).all(), f"{backend} P >= 0 failed")
        
    def test_dual_ascent_all(self):
         for backend in ['torch', 'triton', 'cuda']:
            self._test_dual_ascent(backend)
            
    def _test_routing(self, backend):
        print(f"Testing Routing [{backend}]")
        B, T, E = 2, 10, 5
        logits = torch.randn(B, T, E, device=self.device)
        
        try:
            P = max_score_routing(logits, capacity_factor=1.0, epsilon=0.1, num_iters=20, backend=backend)
        except Exception as e:
            if "not available" in str(e):
                print(f"Skipping {backend}: {e}")
                return
            raise e
            
        row_sums = P.sum(dim=2)
        max_diff = (row_sums - 1.0).abs().max().item()
        print(f"  [{backend}] Row Sum Diff: {max_diff:.6f}")
        self.assertTrue(max_diff < 1e-3, f"{backend} Routing sum!=1")
        
    def test_routing_all(self):
        for backend in ['torch', 'triton', 'cuda']:
            self._test_routing(backend)

if __name__ == '__main__':
    unittest.main()
