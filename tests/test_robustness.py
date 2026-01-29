
import pytest
import torch
import torch_linear_assignment
from efficient_linear_assignment.api import linear_assignment, BACKENDS

# Skip cutile
BACKENDS_TO_TEST = [b for b in BACKENDS if b != 'cutile']

@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
@pytest.mark.parametrize("N", [128, 136, 144, 256]) # Multiples of 8
def test_correctness_vs_hungarian(backend, N):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if backend in ['triton', 'cpp'] and device == 'cpu':
        pytest.skip("Backend requires CUDA")

    B = 2
    M = N # Square
    cost = torch.rand((B, N, M), device=device, dtype=torch.float32)

    # Auction Solver
    # Returns (B, N) one-hot or assignment? 
    # default is one-hot (B, N, N). 
    # Use return_indices=True for easier cost comparison
    auction_indices = linear_assignment(cost, backend=backend, max_iter=5000, epsilon=1e-3, return_indices=True)
    
    # Hungarian Solver (torch_linear_assignment)
    # Expected API: batch_linear_assignment(cost) -> indices (B, N)
    # Note: torch_linear_assignment expects cost to minimize? 
    # Yes, typically. My solver also minimizes cost (internally uses -cost for benefits).
    hungarian_indices = torch_linear_assignment.batch_linear_assignment(cost)
    
    # Compare Costs
    # Total Cost = sum(Cost[b, i, assigned_j])
    
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    agent_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    
    auction_cost = cost[batch_idx, agent_idx, auction_indices].sum(dim=1)
    hungarian_cost = cost[batch_idx, agent_idx, hungarian_indices].sum(dim=1)
    
    # Check if Auction is within bound
    # Bound: TotalCost_Auction <= TotalCost_Hungarian + N * epsilon
    # Actually since it's maximization of benefit:
    # Benefit_Auction >= Benefit_Opt - N*epsilon
    # -Cost_Auction >= -Cost_Opt - N*epsilon
    # Cost_Auction <= Cost_Opt + N*epsilon
    
    epsilon = 1e-3
    tolerance = N * epsilon
    
    diff = auction_cost - hungarian_cost
    
    # Auction typically gives strictly worse (higher) cost than optimal Hungarian.
    # diff should be >= 0 (optimal is minimal)
    # and diff <= tolerance
    
    print(f"\n({backend} N={N}) Auction: {auction_cost.mean():.4f}, Hungarian: {hungarian_cost.mean():.4f}, Diff: {diff.mean():.4f}, Tol: {tolerance:.4f}")

    assert (diff <= tolerance * 1.5).all(), f"Auction failed optimality check. Diff: {diff}, Tol: {tolerance}"


@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
def test_degenerate_cases(backend):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if backend in ['triton', 'cpp'] and device == 'cpu':
        pytest.skip("Backend requires CUDA")
        
    B, N, M = 2, 32, 32
    
    # Case 1: All Zeros
    cost_zeros = torch.zeros((B, N, M), device=device)
    # Any assignment is optimal (Cost=0). Auction should terminate.
    # Logic: benefits=0. prices will stay 0? 
    # Increments might be epsilon.
    indices = linear_assignment(cost_zeros, backend=backend, max_iter=1000, return_indices=True)
    # Check valid permutation
    for b in range(B):
        u = torch.unique(indices[b])
        if len(u) != N:
            print(f"DEBUG: Backend {backend} zero-cost failure. Batch {b}, Unique: {len(u)}, Missing: {N - len(u)}")
            print(f"Indices: {indices[b]}")
        assert len(u) == N, "Assignment must be valid permutation"
        
    # Case 2: All Same Value
    cost_same = torch.full((B, N, M), 10.0, device=device)
    indices = linear_assignment(cost_same, backend=backend, max_iter=1000, return_indices=True)
    for b in range(B):
        u = torch.unique(indices[b])
        if len(u) != N:
            print(f"DEBUG: Backend {backend} flat-cost failure. Batch {b}, Unique: {len(u)}")
            print(f"Indices: {indices[b]}")
        assert len(u) == N
        
    # Case 3: Inf?
    # Ideally inputs shouldn't be Inf, but solver should handle or fail gracefully.
    # Auction calculates Val - Price. -Inf - Price = -Inf.
    # If all -Inf, no bid.
    # Not testing Inf strictly as it violates assumptions often.

@pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
def test_batched_correctness(backend):
    """Ensure batch independent processing."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if backend in ['triton', 'cpp'] and device == 'cpu':
        pytest.skip("Backend requires CUDA")
        
    B, N, M = 4, 32, 32
    cost = torch.rand((B, N, M), device=device)
    
    # Make batch 0 very easy (diagonal lower cost)
    # Make batch 1 very hard (random)
    cost[0] = 100.0
    cost[0].diagonal(dim1=0, dim2=1).copy_(torch.zeros(N, device=device))  # Diagonal is 0, rest 100
    
    indices = linear_assignment(cost, backend=backend, max_iter=2000, return_indices=True)
    
    # Batch 0 should match diagonal exactly (indices 0..N-1)
    expected_b0 = torch.arange(N, device=device)
    # Sort just in case? No, row i assigned to col j.
    # Optimal is (0,0), (1,1)...
    assert torch.equal(indices[0], expected_b0), "Batch 0 should perfectly match diagonal preference"
    
    # Check validity of others
    for b in range(1, B):
        assert len(torch.unique(indices[b])) == N


