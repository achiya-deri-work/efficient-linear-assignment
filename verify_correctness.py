
import torch
import time
from efficient_linear_assignment.api import linear_assignment
from efficient_linear_assignment.utils import pad_input

def verify():
    # Setup
    device = 'cuda'
    current_device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(current_device)
    print(f"Running on {props.name}")
    
    # 1. Deterministic Case (Small)
    print("\n--- Test 1: Small Deterministic Matrix ---")
    # Cost where diag is best
    B, N, M = 2, 32, 32
    cost = torch.randn(B, N, M, device=device).abs()
    # Make diagonal clearly best (0 cost, others > 1)
    # cost[:, range(N), range(N)] = -10.0 # High benefit
    # Actually, benefit = -cost.
    # To prefer diagonal, cost=0, others=10.
    cost = torch.full((B, N, M), 10.0, device=device)
    for b in range(B):
        cost[b].fill_diagonal_(0.0)
        
    print("Solving with Torch...")
    tr_start = time.time()
    assign_torch = linear_assignment(cost, backend='torch', return_indices=True)
    tr_end = time.time()
    
    print("Solving with Triton (Atomic Phase 1)...")
    tt_start = time.time()
    try:
        assign_triton = linear_assignment(cost, backend='triton', return_indices=True)
    except Exception as e:
        print(f"Triton Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    tt_end = time.time()
    
    print(f"Torch Time: {(tr_end-tr_start)*1000:.2f}ms")
    print(f"Triton Time: {(tt_end-tt_start)*1000:.2f}ms")
    
    # Check correctness
    # In this case, indices should be 0..31
    expected = torch.arange(N, device=device).expand(B, N)
    
    match_torch = (assign_torch == expected).all()
    match_triton = (assign_triton == expected).all()
    
    print(f"Torch Correct: {match_torch}")
    print(f"Triton Correct: {match_triton}")
    
    if not match_triton:
        print(f"Triton Result Sample: {assign_triton[0, :10]}")
        
    if not match_triton:
        print(f"Triton Result Sample: {assign_triton[0, :10]}")

    print("Solving with C++ (Small)...")
    try:
        assign_cpp_small = linear_assignment(cost, backend='cpp')
        match_cpp = (assign_cpp_small == expected).all()
        print(f"C++ Correct: {match_cpp}")
        if not match_cpp:
             print(f"C++ Result: {assign_cpp_small[0, :10]}")
    except Exception as e:
        print(f"C++ Small Failed: {e}")

    # 2. Random Case (Large)
    print("\n--- Test 2: Large Random Matrix (N=1024) ---")
    B, N, M = 1, 1024, 1024
    cost = torch.rand(B, N, M, device=device).to(torch.float16) # Test Mixed Precision too
    
    print("Solving Torch...")
    torch.cuda.synchronize()
    t0 = time.time()
    assign_torch = linear_assignment(cost, backend='torch')
    torch.cuda.synchronize()
    t1 = time.time()
    
    print("Solving Triton...")
    torch.cuda.synchronize()
    t2 = time.time()
    assign_triton = linear_assignment(cost, backend='triton')
    torch.cuda.synchronize()
    t3 = time.time()
    
    print(f"Torch: {(t1-t0)*1000:.2f}ms")
    print(f"Triton: {(t3-t2)*1000:.2f}ms")
    
    # Compare Costs using FP32 accumulation
    # indices: (B, N)
    # gather cost: cost[b, i, assign[b, i]]
    
    def get_total_cost(assign, c):
        c_f32 = c.to(torch.float32)
        total = 0.0
        for b_idx in range(B):
            row_idx = torch.arange(N, device=c.device)
            col_idx = assign[b_idx]
            vals = c_f32[b_idx, row_idx, col_idx]
            total += vals.sum().item()
        return total
        
    cost_torch = get_total_cost(assign_torch, cost)
    cost_triton = get_total_cost(assign_triton, cost)
    
    print(f"Total Cost Torch: {cost_torch:.4f}")
    print(f"Total Cost Triton: {cost_triton:.4f}")
    
    diff = abs(cost_torch - cost_triton)
    rel_diff = diff / (abs(cost_torch) + 1e-6)
    
    print(f"Diff: {diff:.4f}, Rel: {rel_diff:.6f}")
    
    print(f"Diff: {diff:.4f}, Rel: {rel_diff:.6f}")
    
    if rel_diff < 0.01: # 1% tolerance
        print("PASS: Costs match (Triton).")
    else:
        print("FAIL: Cost divergence (Triton).")
        
    # 3. C++ Verification
    print("\n--- Test 3: C++ Backend Verification ---")
    print("Solving with C++...")
    try:
        torch.cuda.synchronize()
        t4 = time.time()
        assign_cpp = linear_assignment(cost, backend='cpp')
        torch.cuda.synchronize()
        t5 = time.time()
        print(f"C++ Time: {(t5-t4)*1000:.2f}ms")
        
        cost_cpp = get_total_cost(assign_cpp, cost)
        print(f"Total Cost C++: {cost_cpp:.4f}")
        
        diff_cpp = abs(cost_torch - cost_cpp)
        rel_diff_cpp = diff_cpp / (abs(cost_torch) + 1e-6)
        
        print(f"Diff C++: {diff_cpp:.4f}, Rel: {rel_diff_cpp:.6f}")
        
        if rel_diff_cpp < 0.01:
            print("PASS: Costs match (C++).")
        else:
            print("FAIL: Cost divergence (C++).")
            
        # Optional: Check assignment identity?
        # Assignments might differ if multiple optimal solutions.
        # Cost is the robust metric.
        
    except Exception as e:
         print(f"C++ Failed: {e}")
         import traceback
         traceback.print_exc()

if __name__ == "__main__":
    verify()
