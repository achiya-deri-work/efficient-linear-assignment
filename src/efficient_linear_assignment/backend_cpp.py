import torch
from typing import Tuple
from .backend_torch import AuctionTorch

# Try import extension
HAS_CPP = False
try:
    from . import efficient_linear_assignment_cpp
    HAS_CPP = True
except ImportError as e:
    print(f"DEBUG: First import failed: {e}")
    try:
        import efficient_linear_assignment.efficient_linear_assignment_cpp as efficient_linear_assignment_cpp
        HAS_CPP = True
    except ImportError as e2:
        print(f"DEBUG: Second import failed: {e2}")

class AuctionCPPCCUDA(AuctionTorch):
    def __init__(self, epsilon: float = 1e-2, max_iter: int = 1000):
        super().__init__(epsilon, max_iter)
        if not HAS_CPP:
            raise ImportError("C++ extension not compiled or found.")

    def solve(self, cost_matrix: torch.Tensor, persistent_mode=True) -> Tuple[torch.Tensor, torch.Tensor]:
        if not HAS_CPP:
            raise RuntimeError("C++ extension not available")
            
        # The C++ backend now implements the full loop in 'solve_auction_cuda'
        if hasattr(efficient_linear_assignment_cpp, "solve_auction_cuda"):
            # New Path: Pure C++ Loop
            res = efficient_linear_assignment_cpp.solve_auction_cuda(
                cost_matrix, 
                self.epsilon, 
                self.max_iter,
                persistent_mode
            )
            # res[0] is assignment, res[1] is prices
            return res[0], res[1]
        else:
             raise RuntimeError("Analysis Error: solve_auction_cuda not found in extension. Please rebuild.")
