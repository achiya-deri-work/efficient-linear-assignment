import torch
import pytest
from efficient_linear_assignment.api import linear_assignment

def test_compile_torch_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed for compile test meaningfulness")
        
    cost = torch.rand((2, 16, 16), device='cuda')
    
    # Compile the function
    # Note: backend='torch' uses pure pytorch ops, so it should be fully traceable 
    # except for the loop. 'fullgraph=True' might fail due to the loop (data dependent control flow).
    # But default compile should handle it via graph break or loop unrolling if static.
    # The loop condition `is_unassigned.any()` is data dependent.
    # So we expect a Graph Break.
    
    opt_fn = torch.compile(linear_assignment, backend="inductor")
    
    # Run
    res = opt_fn(cost, backend='torch', return_indices=True)
    assert res.shape == (2, 16)
    
    # Check if consistent
    res2 = linear_assignment(cost, backend='torch', return_indices=True)
    assert torch.allclose(res.float(), res2.float())

def test_compile_triton_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
        
    cost = torch.rand((2, 16, 16), device='cuda')
    
    # Compile
    opt_fn = torch.compile(linear_assignment, mode="reduce-overhead")
    
    # The Triton backend calls a custom Triton kernel.
    # Torch compile handles custom ops generally OK if they have FakeTensor support?
    # Or strict graph breaks.
    
    res = opt_fn(cost, backend='triton')
    assert res.shape == (2, 16)
