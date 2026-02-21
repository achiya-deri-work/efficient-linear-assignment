
import torch
import efficient_linear_assignment.efficient_linear_assignment_cpp as cpp

def solve(Q, K, epsilon=1.0, max_iter=100, math_mode="auto"):
    """
    Solve Dual Ascent using Flash Kernels.
    
    Args:
        Q: [B, N, D] Query tensor.
        K: [B, M, D] Key tensor.
        epsilon: regularization strength.
        max_iter: Max iterations.
        math_mode: 
            - "auto": Selects based on input dtype.
            - "tf32": Force TF32 math (requires Float input).
            - "fp16": Force FP16 math (Tensor Core).
            - "bf16": Force BF16 math (Tensor Core).
    """
    B, N, D = Q.shape
    M = K.shape[1]
    
    mu = torch.ones(B, N, device=Q.device, dtype=Q.dtype) / N
    nu = torch.ones(B, M, device=Q.device, dtype=Q.dtype) / M

    if math_mode == "auto":
        # Default policy:
        # Float -> TF32 (V4)
        # Half -> FP16 (V3)
        # BFloat16 -> BF16
        if Q.dtype == torch.float32:
            math_mode = "tf32"
        elif Q.dtype == torch.float16:
            math_mode = "fp16" 
        elif Q.dtype == torch.bfloat16:
            math_mode = "bf16"
        else:
            raise ValueError(f"Unsupported dtype for auto math_mode: {Q.dtype}")

    res = cpp.flash_dual_ascent_dispatch(Q, K, mu, nu, epsilon, max_iter, math_mode)
    return res[0], res[1]

def check_stability(B=1, N=1024, D=64, dtypes=[torch.float32, torch.float16, torch.bfloat16]):
    """
    Run a brute-force stability check across all supported configurations.
    """
    print(f"--- Flash Dual Ascent Stability Check (N={N}) ---")
    modes = ["tf32", "fp16", "bf16"]
    
    for dtype in dtypes:
        print(f"\n[Input: {dtype}]")
        Q = torch.randn(B, N, D, device='cuda', dtype=dtype)
        K = torch.randn(B, N, D, device='cuda', dtype=dtype) # M=N
        
        for mode in modes:
            print(f"  > Math Mode: {mode:<6} ... ", end="")
            try:
                alpha, beta = solve(Q, K, math_mode=mode)
                
                # Check NaNs
                if torch.isnan(alpha).any() or torch.isnan(beta).any():
                    print("FAIL (NaN Detected)")
                elif torch.isinf(alpha).any() or torch.isinf(beta).any():
                    print("FAIL (Inf Detected)")
                else:
                    # Check Objective roughly
                    # We can't easily check objective without computing C, possibly OOM or precision weirdness
                    # But if no NaNs, it's 'Stable'.
                    print(f"PASS (Mean Alpha: {alpha.mean().item():.3f})")
            except Exception as e:
                print(f"ERROR ({e})")

