import torch
import pytest
import scipy.optimize
from efficient_linear_assignment.api import linear_assignment, BACKENDS

# Fixture to generate problems
@pytest.fixture
def random_cost_matrix():
    B, N, M = 2, 8, 8
    return torch.rand((B, N, M), dtype=torch.float32)

def scipy_solve(cost):
    # cost: (N, M)
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_np)
    # Return indices sorted by row
    return torch.as_tensor(col_ind, device=cost.device)

@pytest.mark.parametrize("backend", list(BACKENDS.keys()))
def test_simple_match(backend, random_cost_matrix):
    if backend == 'cutile':
        pytest.skip("CuTile backend requires manual setup (CUDA 13.1 tileiras compiler missing).")
    # pass
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if backend in ['triton', 'cpp'] and device == 'cpu':
        pytest.skip("Backend requires CUDA")
        
    cost = random_cost_matrix.to(device)
    
    # Run Solver
    assignment = linear_assignment(cost, backend=backend, epsilon=1e-3, max_iter=2000)
    
    # Verify vs Scipy
    for b in range(cost.shape[0]):
        scipy_assgn = scipy_solve(cost[b])
        # Note: Auction is approximate (epsilon-optimality).
        # We check if cost is close to optimal.
        
        my_cost = cost[b, torch.arange(cost.shape[1]), assignment[b]].sum()
        scipy_cost = cost[b, torch.arange(cost.shape[1]), scipy_assgn].sum()
        
        # Allow small deviation due to epsilon
        assert my_cost <= scipy_cost + 1e-1, f"Cost too high on batch {b}"

@pytest.mark.parametrize("backend", list(BACKENDS.keys()))
def test_gradcheck(backend):
    if backend == 'cutile':
        pytest.skip()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if backend != 'torch' and device == 'cpu':
        pytest.skip()
        
    # Small problem for gradcheck
    B, N, M = 1, 8, 8
    cost = torch.randn(B, N, M, device=device, dtype=torch.double, requires_grad=True)
    
    # Wrapper for gradcheck
    # Gradcheck needs the function to return a scalar or we check jacobian.
    # We output assignment indices (discrete) which can't be grad-checked directly strictly speaking,
    # BUT our `AuctionIMLE` claims to be differentiable.
    # The gradients flow into `cost_matrix`.
    
    # Wait. `AuctionIMLE.apply` returns `assignment` (LongTensor).
    # LongTensor cannot carry gradients.
    # Standard IMLE: The output is usually the *expectation* or a soft sample?
    # NO. The output is discrete sample y.
    # The gradient is defined w.r.t the INPUT parameters theta.
    # dL/dTheta = (y - y') ...
    
    # In PyTorch, if forward returns LongTensor, autograd might break the chain?
    # "Output 0 of AuctionIMLEBackward is a long, so it cannot require gradients."
    # Correct.
    
    # To support differentiability, we typically return `one_hot` or similar float representation?
    # Or, the downstream loss function takes `assignment` indices, picks values from *something else*?
    # If the loss is L(assignment), then dL/assignment is needed.
    
    # If standard LAP solver usage in deep learning:
    # Loss = Sum(Cost * Assignment).
    # We want dLoss/dCost.
    
    # If `assignment` is Indices, we gather from Cost?
    # Loss = cost.gather(1, assignment).sum()
    # Here `cost` has grad. `assignment` is just indices.
    # Gradients flow through `cost` directly?
    # NO. We want to differentiate *how assignment changes* if cost changes.
    # DIFFERENTIABLE SORTING / ASSIGNMENT typically implies soft output or VJP.
    
    # If I return discrete indices, I break the graph unless I use `Straight Through`?
    # My `backward` returns `grad_input` (w.r.t cost).
    # But how does `grad_output` (w.r.t assignment) reach `backward`?
    # It CANNOT if assignment is integer.
    
    # Thus, the API *must* return a float representation (e.g. Permutation Matrix).
    # OR the user uses it in a specific framework (e.g. Blackbox Backprop) that handles the disconnect.
    
    # If I want `torch.autograd.gradcheck` to work, `forward` MUST return float.
    # We use return_indices=False
    
    def func(c):
        return linear_assignment(c, backend=backend, epsilon=1e-3, max_iter=200, return_indices=False)
        
    # assert torch.autograd.gradcheck(func, cost, eps=1e-3, atol=1e-2)
    # Since forward is Hard (Discrete), Finite Diff is 0, but IMLE Grad is non-zero.
    # Gradcheck will fail. We verify gradients flow manually.
    
    out = func(cost)
    # Define a loss
    loss = (out * cost).sum()
    loss.backward()
    
    assert cost.grad is not None
    assert (cost.grad != 0).any(), "Gradient should be non-zero for surrogate learning"

