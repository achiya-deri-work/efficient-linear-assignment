
import torch
import time
from efficient_linear_assignment.utils import AsyncLinearAssignment
from efficient_linear_assignment.auction import linear_assignment

def test_async_solver():
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available")
        return

    print("=== Testing Async Solver ===")
    
    # CPU Input
    B, N, M = 4, 256, 256
    cost_cpu = torch.randn(B, N, M)
    
    solver = AsyncLinearAssignment()
    
    # Warmup
    future = solver.submit(linear_assignment, cost_cpu)
    _ = future.get()
    
    torch.cuda.synchronize()
    
    # Timed Run
    t0 = time.time()
    future = solver.submit(linear_assignment, cost_cpu)
    t1 = time.time()
    
    submit_duration = t1 - t0
    print(f"Submit Duration (Non-blocking): {submit_duration*1000:.3f} ms")
    
    # Simulate work
    time.sleep(0.05)
    
    # Get Result
    t2 = time.time()
    result = future.get()
    t3 = time.time()
    get_duration = t3 - t2
    
    print(f"Get Duration (Wait): {get_duration*1000:.3f} ms")
    print(f"Result Device: {result.device}")
    
    assert result.shape == (B, N)
    assert result.device.type == 'cuda'
    print("[PASS] Async Solver functionality verified.")

if __name__ == "__main__":
    test_async_solver()
