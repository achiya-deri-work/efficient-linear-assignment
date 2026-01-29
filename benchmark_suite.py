
import torch
import time
import pandas as pd
import json
import gc
from efficient_linear_assignment.api import linear_assignment
try:
    import torch_linear_assignment
    HAS_EXTERNAL = True
except ImportError:
    HAS_EXTERNAL = False
    print("WARNING: torch_linear_assignment not found. Skipping external benchmark.")

# Configuration
SIZES = [128, 256, 512, 1024, 2048, 4096] # 8192 might OOM on smaller GPUs or take long
BATCH_SIZES = [1] 
# Distinct backends for benchmarking
BACKENDS = ['torch', 'triton', 'cpp_legacy', 'cpp_persistent']
if HAS_EXTERNAL:
    BACKENDS.append('external')

try:
    from efficient_linear_assignment.backend_cpp import AuctionCPPCCUDA
except ImportError:
    print("Could not import AuctionCPPCCUDA. C++ backends will fail.")

results = []

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    for N in SIZES:
        for B in BATCH_SIZES:
            M = N
            print(f"\n--- Config: B={B}, N={N} ---")
            
            # Data Gen
            cost = torch.rand((B, N, M), device=device, dtype=torch.float32)
            
            for backend in BACKENDS:
                # Skip external for very large
                if backend == 'external' and N > 2048:
                    continue
                
                try:
                    stmt = None
                    # Custom handling for C++ modes to toggle flag
                    if backend == 'cpp_legacy':
                        solver = AuctionCPPCCUDA(max_iter=5000)
                        stmt = lambda: solver.solve(cost, persistent_mode=False)
                    elif backend == 'cpp_persistent':
                        solver = AuctionCPPCCUDA(max_iter=5000)
                        stmt = lambda: solver.solve(cost, persistent_mode=True)
                    elif backend == 'external':
                        stmt = lambda: torch_linear_assignment.batch_linear_assignment(cost)
                    else:
                        # Torch, Triton via API
                        stmt = lambda: linear_assignment(cost, backend=backend, max_iter=5000)
                    
                    # Warmup
                    for _ in range(2):
                        stmt()
                    torch.cuda.synchronize()
                    
                    # Measure Time
                    t0 = time.time()
                    if backend == 'torch':
                        ITER = 1
                    else:
                        ITER = 20 if N <= 256 else (10 if N <= 1024 else 5)
                        
                    for _ in range(ITER):
                        stmt()
                    torch.cuda.synchronize()
                    t1 = time.time()
                    avg_time_ms = ((t1 - t0) / ITER) * 1000
                    
                    # Measure Memory
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    mem_start = torch.cuda.memory_allocated()
                    stmt()
                    mem_peak = torch.cuda.max_memory_allocated()
                    mem_usage_mb = (mem_peak - mem_start) / (1024 * 1024)
                    
                    print(f"{backend}: {avg_time_ms:.3f} ms | {mem_usage_mb:.2f} MB")
                    
                    results.append({
                        "Backend": backend,
                        "Batch": B,
                        "Size": N,
                        "Time_ms": avg_time_ms,
                        "Memory_MB": mem_usage_mb
                    })
                    
                except Exception as e:
                    print(f"{backend} Failed: {e}")
                
                # Cleanup
                gc.collect()
                torch.cuda.empty_cache()

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults saved to benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
