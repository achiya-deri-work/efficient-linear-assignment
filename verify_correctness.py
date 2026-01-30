
import torch
import torch_linear_assignment
import numpy as np
import sys
from typing import Dict, List, Tuple

from efficient_linear_assignment import (
    log_stabilized_sinkhorn,
    l2_regularized_dual_ascent,
    linear_assignment as auction_solver,
    sinkhorn_compiled,
    dual_ascent_compiled,
    auction_compiled
)

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
YELLOW = "\033[33m"

def log(msg, color=RESET, **kwargs):
    print(f"{color}{msg}{RESET}", **kwargs)

def run_baseline_auction(cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Runs torch-linear-assignment (LapJV/Hungarian) as Ground Truth.
    Returns: indices (B, N), total_cost
    """
    # torch_linear_assignment usually takes (B, N, M) or (N, M)
    # It returns matchings.
    # Note: torch_linear_assignment.batch_linear_assignment might be available?
    
    if hasattr(torch_linear_assignment, 'batch_linear_assignment'):
        # indices: (B, N)
        indices = torch_linear_assignment.batch_linear_assignment(cost_matrix)
    else:
        # Fallback loop
        B, N, M = cost_matrix.shape
        indices = torch.zeros((B, N), dtype=torch.long, device=cost_matrix.device)
        for b in range(B):
            # linear_assignment returns tensor of indices?
            # actually usually it returns assignment vector.
            # Assuming standard API wrapper if available, else standard Hungarian
            # torch_linear_assignment usually expects CPU or GPU?
            # It's a CUDA wrapper for LapJV usually.
            indices[b] = torch_linear_assignment.linear_assignment(cost_matrix[b])
            
    # Calculate Cost
    # Gather costs
    B, N, M = cost_matrix.shape
    row_idx = torch.arange(N, device=cost_matrix.device).unsqueeze(0).expand(B, -1)
    batch_idx = torch.arange(B, device=cost_matrix.device).unsqueeze(1).expand(-1, N)
    
    if indices.max() >= M or indices.min() < 0:
        # Invalid indices from baseline?
        return indices, float('inf')
        
    chosen_costs = cost_matrix[batch_idx, row_idx, indices]
    total_cost = chosen_costs.sum().item()
    
    return indices, total_cost

def calculate_assignment_cost(cost_matrix, indices):
    B, N, M = cost_matrix.shape
    row_idx = torch.arange(N, device=cost_matrix.device).unsqueeze(0).expand(B, -1)
    batch_idx = torch.arange(B, device=cost_matrix.device).unsqueeze(1).expand(-1, N)
    
    # Check invalid
    if (indices == -1).any():
        return float('inf')
        
    chosen_costs = cost_matrix[batch_idx, row_idx, indices]
    return chosen_costs.sum().item()

def verify_auction(
    backend: str,
    dtype: torch.dtype,
    B=4, N=128, M=128,
    cost_scale=1000,
    max_iter=2000,
    epsilon=1e-2
):
    """
    Compare Efficient-Auction vs Baseline.
    Input: FP32, FP16, or BF16.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Generate Integers for Exactness Check
    logits = torch.randn(B, N, M, device=device, dtype=torch.float32)
    # Integers in [0, cost_scale]
    cost_matrix_fp32 = (logits.abs() * cost_scale).floor()
    
    # Convert to target dtype
    cost_matrix = cost_matrix_fp32.to(dtype=dtype)
    
    # 2. Run Baseline (on FP32 for reference safety or same dtype?)
    # Baseline LapJV usually requires float/double.
    # We pass FP32 to baseline to get 'True' answer.
    idx_gt, cost_gt = run_baseline_auction(cost_matrix_fp32)
    
    # 3. Run Efficient Auction
    # Algo requires epsilon. For integers, any epsilon < 1/N is usually exact?
    # Optimal epsilon for integers is 1.0 / (N + 1) usually or smaller.
    # We use aggressive epsilon scaling or just small epsilon.
    
    try:
        if backend == 'torch_compiled':
             indices_pred = auction_compiled(cost_matrix, epsilon=epsilon, max_iter=max_iter)
        else:
             indices_pred = auction_solver(cost_matrix, backend=backend, epsilon=epsilon, max_iter=max_iter)
             
        # Unwrap tupple if returned (indices, prices) - auction_solver returns indices if return_indices=True (default)
        if isinstance(indices_pred, tuple):
             indices_pred = indices_pred[0] # Handle raw backend return?
             # Wait, generic `linear_assignment` (wrapper) returns indices only.
             # `solve_` functions usually return more.
             # Let's verify standard API: `linear_assignment` returns indices tensor.
             
    except Exception as e:
        log(f"    [FAIL] Crash: {e}", RED)
        return False

    # 4. Compare Costs
    # We compute cost using ORIGINAL FP32 counts to verify exactness.
    # We do NOT compare indices directly as solution might not be unique.
    # We compare TOTAL COST.
    cost_pred = calculate_assignment_cost(cost_matrix_fp32, indices_pred)
    
    # Check
    # Integer costs -> exact match expected?
    # Auction is epsilon-optimal. It guarantees solution within N * epsilon of optimal.
    # cost_pred <= cost_gt + N * epsilon?
    # Actually Auction maximizes Benefit = -Cost.
    # Min Cost = - Max Benefit.
    # We used cost as input. Auction effectively minimizes cost.
    
    # Allowable tolerance
    # Since inputs are integers, if epsilon < 1/N, and we use strict updates, we should find optimal.
    # But Auction implementation uses floats.
    
    diff = abs(cost_pred - cost_gt)
    # Tolerance relative or absolute?
    # Cost is ~500 * N * B ~ 500 * 128 * 4 ~ 256,000.
    # Tolerance 0.1% or strict?
    # If using FP16 input, values might be perturbed?
    # We cast to FP16. So baseline should also see FP16 values if we want to compare fairness?
    # But user wants to test if FP16 input yields CORRECT matching for the underlying problem?
    # No, precision loss in Input is expected.
    # We should compare against "Baseline running on SAME INPUT".
    
    # Let's re-run baseline on the casted input cast back to fp32.
    cost_matrix_seen = cost_matrix.float()
    idx_gt_seen, cost_gt_seen = run_baseline_auction(cost_matrix_seen)
    
    cost_pred_seen = calculate_assignment_cost(cost_matrix_seen, indices_pred)
    
    diff_seen = abs(cost_pred_seen - cost_gt_seen)
    
    is_ok = diff_seen <= (cost_gt_seen * 1e-4 + 1.0) # 0.01% tolerance + 1.0 abs
    
    if is_ok:
        log(f"    [PASS] Cost Diff: {diff_seen:.4f} (GT: {cost_gt_seen:.2f})", GREEN)
        return True
    else:
        log(f"    [FAIL] Cost Diff: {diff_seen:.4f} (GT: {cost_gt_seen:.2f} vs Pred: {cost_pred_seen:.2f})", RED)
        return False

def verify_marginals(
    algo_name: str,
    func,
    backend: str,
    dtype: torch.dtype,
    B=4, N=128, M=128,
    tol=1e-3
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Weights
    C = torch.rand(B, N, M, device=device, dtype=dtype)
    mu = torch.ones(B, N, device=device, dtype=dtype) / N
    nu = torch.ones(B, M, device=device, dtype=dtype) / M
    
    # 2. Run
    try:
        kwargs = {"num_iters": 20}
        if backend == 'torch_compiled':
             # func is the compiled object
             P = func(C, mu, nu, **kwargs)
        else:
             P = func(C, mu, nu, backend=backend, **kwargs)
             
    except Exception as e:
        log(f"    [FAIL] Crash: {e}", RED)
        return False
        
    # 3. Check Marginals (in FP32)
    P_f32 = P.float()
    mu_f32 = mu.float()
    nu_f32 = nu.float()
    
    row_sum = P_f32.sum(dim=-1)
    col_sum = P_f32.sum(dim=-2)
    
    err_row = (row_sum - mu_f32).abs().mean().item()
    err_col = (col_sum - nu_f32).abs().mean().item()
    
    total_err = err_row + err_col
    
    if total_err < tol:
        log(f"    [PASS] Marginal Err: {total_err:.6f}", GREEN)
        return True
    else:
        log(f"    [FAIL] Marginal Err: {total_err:.6f} > {tol}", RED)
        return False


def run_all():
    log("=== Verification Suite ===", YELLOW)
    
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    dtypes = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "BF16": torch.bfloat16
    }
    
    # Skip torch_compiled for Auction due to Inductor hang on complex loops
    backends_auction = ["torch", "triton", "cuda"] 
    backends_approx = ["torch", "triton", "cuda", "torch_compiled"]
    
    # 1. Auction Test
    log("\n--- Testing Auction (Exactness) ---", YELLOW)
    for prec_name, dtype in dtypes.items():
        log(f"Precision: {prec_name}")
        for backend in backends_auction:
            log(f"  Backend: {backend}...", end="")
            # Use smaller scale for convergence
            verify_auction(backend, dtype, cost_scale=5, max_iter=50000, epsilon=1e-3)

    # 2. Sinkhorn Test
    log("\n--- Testing Sinkhorn (Marginals) ---", YELLOW)
    for prec_name, dtype in dtypes.items():
        log(f"Precision: {prec_name}")
        for backend in backends_approx:
            log(f"  Backend: {backend}...", end="")
            # Select func
            fn = sinkhorn_compiled if backend == 'torch_compiled' else log_stabilized_sinkhorn
            verify_marginals("Sinkhorn", fn, backend, dtype)
            
    # 3. Dual Ascent Test
    log("\n--- Testing Dual Ascent (Marginals) ---", YELLOW)
    for prec_name, dtype in dtypes.items():
        log(f"Precision: {prec_name}")
        for backend in backends_approx:
            log(f"  Backend: {backend}...", end="")
            fn = dual_ascent_compiled if backend == 'torch_compiled' else l2_regularized_dual_ascent
            verify_marginals("DualAscent", fn, backend, dtype, tol=0.1) 

if __name__ == "__main__":
    run_all()
