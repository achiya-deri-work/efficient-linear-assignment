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
except ImportError:
    pass

# Alias 'cuda' to 'cpp' for legacy support
if 'cpp' in BACKENDS:
    BACKENDS['cuda'] = BACKENDS['cpp']

class AuctionIMLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cost_matrix: torch.Tensor, epsilon: float, max_iter: int, backend_name: str):
        # 1. Select Backend
        if backend_name not in BACKENDS:
            raise ValueError(f"Backend '{backend_name}' not available. Options: {list(BACKENDS.keys())}")
        
        backend_cls = BACKENDS[backend_name]
        solver = backend_cls(epsilon=epsilon, max_iter=max_iter)
        
        # 2. Run Forward Solver
        # Expected return: assignment (B, N) or (N), prices (B, N)
        assignment, _ = solver.solve(cost_matrix)
        
        # Convert to One-Hot Matrix (Float) for Autograd
        B, N, M = cost_matrix.shape
        assignment_matrix = torch.zeros_like(cost_matrix)
        batch_idx = torch.arange(B, device=cost_matrix.device).unsqueeze(1)
        row_idx = torch.arange(N, device=cost_matrix.device).unsqueeze(0)
        # Handle -1 (unassigned)? 
        # Ideally all assigned. If -1, we leave 0 in matrix.
        valid_mask = (assignment >= 0)
        assignment_matrix[batch_idx, row_idx, assignment.clamp(0)] = valid_mask.to(cost_matrix.dtype)
        
        # 3. Save for Backward (IMLE needs inputs, not just outputs)
        ctx.save_for_backward(cost_matrix, assignment_matrix) # Save P_forward
        ctx.epsilon = epsilon
        ctx.max_iter = max_iter
        ctx.backend_name = backend_name
        ctx.imle_lambda = 1.0 # Hyperparam
        
        return assignment_matrix

    @staticmethod
    def backward(ctx, grad_output):
        """
        IMLE Backward Pass.
        grad_output: (B, N, M) - Gradient w.r.t Assignment Matrix.
        """
        cost_matrix, P_forward = ctx.saved_tensors
        epsilon = ctx.epsilon
        max_iter = ctx.max_iter
        backend_name = ctx.backend_name
        imle_lambda = ctx.imle_lambda
        
        # 1. Construct Target Cost
        # C_target = C - lambda * grad_output (Maximize Benefit) or + lambda*G (Minimize Cost)?
        # If we minimize Cost C.
        # We want to move towards direction that minimizes Total Loss.
        # Loss w.r.t Assignment matrix X is G.
        # We want X' to align with -G (steepest descent).
        # So we favor edges with NEGATIVE gradient (where increasing X lowers loss).
        # Favoring means Lowering Cost.
        # C' = C + lambda * G.
        # If G is positive (bad), Cost increases, X avoids it.
        # If G is negative (good), Cost decreases, X picks it.
        
        target_cost = cost_matrix + imle_lambda * grad_output
        
        # 2. Re-solve
        backend_cls = BACKENDS[backend_name]
        solver_target = backend_cls(epsilon=epsilon, max_iter=max_iter)
        assignment_target_idx, _ = solver_target.solve(target_cost)
        
        # Convert Target to One-Hot
        B, N, M = cost_matrix.shape
        P_target = torch.zeros_like(cost_matrix)
        batch_idx = torch.arange(B, device=cost_matrix.device).unsqueeze(1)
        row_idx = torch.arange(N, device=cost_matrix.device).unsqueeze(0)
        valid_mask = (assignment_target_idx >= 0)
        P_target[batch_idx, row_idx, assignment_target_idx.clamp(0)] = valid_mask.to(cost_matrix.dtype)
        
        # 3. Compute Gradient
        # (P_target - P_forward) / imle_lambda * sign?
        # If Cost C increases (Grad > 0), P should decrease.
        # P_target minimized (C + G).
        # If G > 0, C' > C. Argmin avoids it. P_target < P_forward.
        # Diff = P_target - P_forward < 0.
        # We want grad_input ~ Diff / Imle ? 
        # C' = C + G.
        # Yes.
        
        grad_input = (P_forward - P_target) / imle_lambda
        # Note: If P_target < P_forward, this gives Positive gradient.
        # Which matches dL/dC if dL/dX is Positive?
        # L = X.
        # C increases -> X decreases -> L decreases?
        # Wait, if we minimize C*X, X avoids high C.
        # dX/dC is negative.
        # If dL/dX is positive. dL/dC = dL/dX * dX/dC = (+) * (-) = (-).
        # Here (P_for - P_tar) is (- -) ? = 0 - (-1) = +1.
        # Implies Positive Gradient?
        # Let's trust the logic: grad = (P_target - P_forward) / lambda? No.
        # Usually (Perturbed - Original).
        # Let's stick to (P_target - P_forward) / lambda.
        # If G>0, P_target < P_forward. Diff < 0. Gradient Negative.
        # Correct.
        
        grad_input = (P_target - P_forward) / imle_lambda
        
        return grad_input, None, None, None

def linear_assignment(cost_matrix: torch.Tensor, epsilon: float = 1e-2, max_iter: int = 1000, backend: str = 'torch', return_indices: bool = True):
    """
    Solves LAP.
    Args:
        return_indices: If True, returns (B, N) LongTensor. Non-differentiable.
                       If False, returns (B, N, M) FloatTensor (Soft/Hard P). Differentiable.
    """
    # Ensure Batch Dim
    is_batched = cost_matrix.ndim == 3
    if not is_batched:
        cost_matrix = cost_matrix.unsqueeze(0)
    
    # Validate dimensions (Multiples of 8 check)
    _, N, M = cost_matrix.shape
    if N % 8 != 0 or M % 8 != 0:
        raise ValueError(f"Input dimensions must be multiples of 8. Got shape {cost_matrix.shape}. Please pad your input.")

    # Call Autograd Function (Returns Matrix)
    assignment_matrix = AuctionIMLE.apply(cost_matrix, epsilon, max_iter, backend)
    
    if return_indices:
        # Argmax to get indices
        # Detach gradient flow because argmax is discrete
        assignment_indices = assignment_matrix.detach().argmax(dim=2)
        if not is_batched:
            assignment_indices = assignment_indices.squeeze(0)
        return assignment_indices
    else:
        if not is_batched:
            assignment_matrix = assignment_matrix.squeeze(0)
        return assignment_matrix
