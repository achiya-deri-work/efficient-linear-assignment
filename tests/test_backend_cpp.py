
import torch
import pytest
import time

def test_cpp_backend_auction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed for C++ verification")

    try:
        from efficient_linear_assignment.auction.cpp_backend import AuctionCPPCCUDA
    except ImportError:
        pytest.skip("C++ Backend not importable")
    except Exception as e:
        pytest.fail(f"Failed to import C++ backend: {e}")

    device = 'cuda'
    
    # 1. Small Deterministic
    B, N, M = 2, 32, 32
    cost = torch.full((B, N, M), 10.0, device=device)
    for b in range(B):
        cost[b].fill_diagonal_(0.0)
        
    expected = torch.arange(N, device=device).expand(B, N)
    
    solver = AuctionCPPCCUDA()
    
    # Test Legacy
    try:
        assign, _ = solver.solve(cost, persistent_mode=False)
        assert (assign == expected).all(), "Legacy C++ solver returned incorrect assignment"
    except Exception as e:
        pytest.fail(f"Legacy Mode Failed: {e}")

    # Test Persistent
    try:
        assign, _ = solver.solve(cost, persistent_mode=True)
        assert (assign == expected).all(), "Persistent C++ solver returned incorrect assignment"
    except Exception as e:
        pytest.fail(f"Persistent Mode Failed: {e}")

    # 2. Large Random Basic Check (Smoke Test)
    # Just ensure it runs without error and returns valid indices
    B, N, M = 1, 128, 128
    cost = torch.rand(B, N, M, device=device)
    
    assign1, _ = solver.solve(cost, persistent_mode=False)
    assign2, _ = solver.solve(cost, persistent_mode=True)
    
    assert assign1.shape == (B, N)
    assert assign2.shape == (B, N)
    
    # Validity check
    assert (assign1 >= -1).all() and (assign1 < M).all()
