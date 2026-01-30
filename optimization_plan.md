# Optimization Plan: Efficient Linear Assignment

## Executive Summary

The current implementation of `efficient-linear-assignment` provides robust explicit-matrix solvers (Auction, Sinkhorn, Dual Ascent) using CUDA, Triton, and Torch. However, performance is largely **Memory Bandwidth Bound** due to the $O(N^2)$ reads of the Cost Matrix $C$ in every iteration.

To achieve "Tensor Core" performance and scale to larger problems, we must shift paradigm from **Explicit Cost Matrices** to **Implicit (Low-Rank) Cost Matrices**, where $C = Q K^T$. This allows computation to become Compute Bound (GEMM), effectively utilizing Tensor Cores.

## 1. Algorithmic Optimizations

### 1.1 "Tensor Core" Activation: Implicit Cost Solvers

**Status**: Not Implemented (Library takes `C` as input).
**Opportunity**: Most deep learning assignment problems invoke $C$ from dot-products (e.g. Attention, DETR matching).
**Plan**:

1.  **API Extension**: Create `solve_implicit(Q, K, ...)` entry points.
2.  **Fused Kernel (Flash-Sinkhorn)**:
    - Load tiles of $Q, K$ into SRAM.
    - Compute tile $C_{block} = Q_{block} K_{block}^T$ using **Tensor Cores (HMMA)**.
    - Apply Sinkhorn/Auction update logic on registers/SRAM.
    - **Benefit**: Reduces Global Memory traffic from $O(N^2)$ per iter to $O(N \cdot D)$. Speedup: **10x-100x** for large $N$.

### 1.2 Epsilon Scaling (Auction/Sinkhorn)

**Status**: Constant epsilon.
**Opportunity**: Annealing epsilon (starting large, decreasing) speeds up convergence significantly.
**Plan**: Implement `epsilon_schedule` generator in Python frontend.

## 2. Backend-Specific Optimizations

### 2.1 Triton Backend

- **Current State**: Loop over $M$ in chunks. Basic vectorization.
- **Optimizations**:
  1.  **Block-Level Reductions**: Use `tl.max` and `tl.sum` with axis arguments rather than manual loops for row reductions.
  2.  **Autotuning**: Expand `triton.autotune` config space to search `num_warps` (4, 8) and `num_stages` (2, 3, 4) for optimal pipelining.
  3.  **Use `tl.dot` for Implicit**: If implementing Implicit Cost, `tl.dot` automatically uses Tensor Cores.

### 2.2 CUDA / CUTLASS Backend

- **Current State**: Persistent Kernels with `cutlass::Array`. 1 Block/Agent for Auction.
- **Optimizations**:
  1.  **Memory Coalescing**: Ensure `stride_bn`, `stride_bm` allow 128-bit vectorized loads. Add `assert` checks or padding logic (Pad $N, M$ to multiples of 8/16).
  2.  **Occupancy Tuning**: Current `auction_persistent` allows 1 block per agent. For small $N$ (e.g., 64), GPU is underutilized.
      - _Fix_: Map multiple small agents to single block (Warp-per-Agent) or use `Cuda Graphs` to batch small problems.
  3.  **BF16/FP16 Math**: Currently casts to float for logic.
      - _Opt_: Use `__hadd2`, `__hfma2` (half2) intrinsics for low-precision-tolerant phases (Dual Ascent P-step).

### 2.3 Torch Backend

- **Current State**: `torch.compile` used, but Newton steps were unstable.
- **Optimizations**:
  1.  **Reduce Overhead**: Avoid `scatter` / `gather` where possible. Use `view` + `bcast`.
  2.  **Kernel Fusion**: _Legacy Note_: `torch.compile` is already active with `max_autotune_gemm` and `aggressive_fusion` in `compiled.py`, providing optimal performance. Focus on algorithm-level fusion (Implicit Costs).

## 3. Data & System Optimizations

### 3.1 Async Dataloading & Pipelining

**Observation**: Assignment often blocks training loop.
**Plan**:

1.  **Non-Blocking Transfer**: Ensure `C` (or $Q, K$) are moved to GPU with `non_blocking=True`.
2.  **CUDA Graphs**:
    - Sinkhorn/Dual Ascent (fixed iter) are Graph-friendly.
    - **Action**: Wrap `solve` in `torch.cuda.make_graphed_callables`.
    - _Constraint_: Dynamic shapes trigger recompilation. Use **Padding** (bucketing) to fixed shapes (e.g. 512, 1024, 2048).

### 3.2 Padding & Alignment

**Issue**: Algorithms perform best when dimensions are multiples of 32/128.
**Plan**: Auto-pad inputs in Python wrapper to next power-of-2 or tile size, then slice output.

## Implementation Roadmap

| Phase | Task                             | Backend | Est. Impact          |
| :---- | :------------------------------- | :------ | :------------------- |
| **1** | **Implicit Cost (Flash-Solver)** | Triton  | **High (TC Active)** |
| **2** | **CUDA Graph Support**           | Python  | Medium (Latency)     |
| **3** | **Autotune Config Expansion**    | Triton  | Low-Medium           |
| **4** | **Async/Non-Blocking Utils**     | Python  | Medium               |
