
import torch
import pytest
from efficient_linear_assignment import sinkhorn_compiled, dual_ascent_compiled

def test_compiled_api_sinkhorn():
    """Verify sinkhorn_compiled API works fundamentally."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    C = torch.randn(2, 64, 64, device=device)
    try:
        res = sinkhorn_compiled(C)
        assert res.shape == (2, 64, 64)
        assert not torch.isnan(res).any()
    except Exception as e:
        if "compile" in str(e) and device == "cpu":
            pytest.skip("Compile might be unsupported on CPU for some configs")
        raise e

def test_compiled_api_dual_ascent():
    """Verify dual_ascent_compiled API works fundamentally."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    C = torch.randn(2, 64, 64, device=device)
    try:
        res = dual_ascent_compiled(C)
        assert res.shape == (2, 64, 64)
        assert not torch.isnan(res).any()
    except Exception as e:
         if "compile" in str(e) and device == "cpu":
            pytest.skip("Compile might be unsupported on CPU for some configs")
         raise e
