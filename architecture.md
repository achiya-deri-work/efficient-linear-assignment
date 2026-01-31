# Efficient Linear Assignment: System Architecture

## 1. High-Level Design

The library is designed as a **Multi-Backend, High-Performance Solver Suite** for Differentiable Linear Assignment. It decouples the _Algorithm_ (Sinkhorn, Dual Ascent, Auction) from the _Execution Engine_ (Torch, Triton, CUDA/Cutlass).

### System Layers

1.  **Frontend (`api.py`)**: Unified entry points (`log_stabilized_sinkhorn`, etc.). Handles input validation and backend dispatch.
2.  **Dispatch Layer (`compiled.py`)**: Smart caching and JIT compilation using `torch.compile` on Inductor.
3.  **Kernel Layer**:
    - **Torch/Inductor**: Fused CUDA kernels generated at runtime.
    - **Triton**: Block-based custom kernels for specific layouts.
    - **Native CUDA/Cutlass**: Pre-compiled C++ binaries for maximum architectural control.

## 2. Core Algorithms

Detailed architecture for each solver is documented separately:

- **[Sinkhorn Architecture](architecture/sinkhorn.md)**: Fast, differentiable approximation using Log-Sum-Exp row/col balancing.
- **[Dual Ascent Architecture](architecture/dual_ascent.md)**: Sparse, L2-regularized solver using Newton Coordinate Descent.
- **[Auction Architecture](architecture/auction.md)**: Exact solver using iterative bidding, optimized with Block-Parallel CUDA kernels.

## 3. Key Optimization Patterns

### A. The "Compiled Dispatcher" (`compiled.py`)

To solve the `torch.compile` "Recompile Limit" instability, we implemented a custom dispatcher:

- **Shape-Specialization**: Creating a fresh function closure for every unique input shape $(B, N, M)$ ensures Inductor optimizes exactly for that size (unrolling loops, fusing constants).
- **Nested Compile Region**: Iterative loops are separated from the inner step. `step = torch.compile(inner)` allows the compiler to fuse the heavy lifting (reductions) without unrolling the entire loop body, keeping compilation times low (<500ms).

### B. Mixed Precision & TF32

- **TF32 (TensorFloat-32)**: Globally enabled for Ampere+ GPUs. Allows FP32 operations to run at near-FP16 speeds by sacrificing mantissa precision, which is acceptable for Transport algorithms.
- **FP16/BF16 Support**: Kernels support Half precision for storage/transfer, casting to Float32 for critical accumulation steps to prevents underflow in Iterative updates.

### C. CUDA vs Torch

Our benchmarks (Phase 7) concluded that **Torch Compiled** is often the superior backend for general use:

- **Zero-Overhead**: No Python-to-C++ dispatch latency.
- **Fusion**: Inductor fuses `Exp -> Sum -> Div` chains better than manual CUDA unless highly tuned.
- **Use Case**: Use `compiled` for $N < 4096$. Use `cutlass` for extreme scales where manual memory coalescing outperforms the compiler.
