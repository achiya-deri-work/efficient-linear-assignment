# Efficient Differentiable Linear Assignment Context

## 1. Project Overview

This library provides high-performance, differentiable solvers for the **Linear Assignment Problem (LAP)**, specifically optimized for GPU execution. It focuses on the **Dual Ascent** algorithm with L2 regularization, which allows for parallelizable, differentiable updates suitable for deep learning pipelines (e.g., matching in DETR, tracking in MOT).

**Key Innovation**: The "**Flash**" kernels (V4), which solve the assignment problem _implicitly_. Instead of materializing the full $N \times M$ cost matrix $C$ in memory (which is $O(N^2)$), we compute $C_{ij} = Q_i \cdot K_j^T$ on-the-fly within the CUDA kernels using Tensor Cores. This reduces memory usage from quadratic to linear $O(ND + MD)$ and significantly increases throughput.

## 2. Core Components

### 2.1 Python API (`src/efficient_linear_assignment`)

- **`flash_dual_ascent`**: The recommended entry point.
  - `solve(Q, K, epsilon, max_iter, math_mode)`: Dispatches to the optimized implicit C++ kernel.
- **`dual_ascent`**: Contains legacy/baseline backends.
  - `torch_backend`: Pure PyTorch implementation (good for reference/debugging).
  - `triton_backend`: Triton implementation (explicit $C$).
  - `cutlass_backend`: Explicit C++ Cutlass implementation.

### 2.2 C++ / CUDA Backend (`src/csrc`)

- **`flash_dual_ascent.cu`**: The monolithic, optimized CUDA kernel.
  - **Unified V3/V4 Implementation**: Combines mixed-precision logic (V3) and TF32 optimization (V4) into a single templated kernel.
  - **Implicit GEMM**: Uses `CuTe` and `TiledMMA` to perform block-wise $Q \cdot K^T$ and dual updates in registers.
  - **Precision Support**: Templated for `InputT` (FP32, FP16, BF16) and `MathT` (TF32, FP16, BF16).
- **`interface.cpp`**: PyBind11 definitions exposing C++ functions to Python.

## 3. Quick Start

### Installation

```bash
# Requires CUDA Toolkit 12.0+ (Tested on 12.8) and PyTorch
pip install .
# Or for mutable dev install
pip install -e .
```

### Usage (Flash Solver)

```python
import torch
import efficient_linear_assignment.flash_dual_ascent as flash

B, N, D = 1, 4096, 64
Q = torch.randn(B, N, D, device='cuda', dtype=torch.float32)
K = torch.randn(B, N, D, device='cuda', dtype=torch.float32)

# Solve for implicit Cost C = -(Q @ K.T)
# Returns dual variables alpha, beta.
alpha, beta = flash.solve(Q, K, epsilon=1.0, max_iter=100)
```

## 4. Development & Testing

### 4.1 Benchmarking

Run the comprehensive benchmark suite to compare all backends:

```bash
export PYTHONPATH=src
python benchmark_all.py
```

_Look for "Flash V4" dominating at N=4096 and N=8192._

### 4.2 Precision Stability

Verify numerical stability across FP16/BF16/TF32 modes:

```bash
python test_precision_stability.py
```

## 5. Architecture Notes

- **Implicit vs Explicit**: Explicit solvers accept $C$ ($N \times N$). Implicit Flash solvers accept embeddings $Q, K$ ($N \times D$). Flash is strictly superior for large $N$.
- **Accumulation**: All Flash kernels enforce **FP32 accumulation** for the dual gradients to prevent numerical instability, even when input/math is FP16.
- **Block sizes**: Optimized for 128x128 blocks using Hopper/Ampere layouts.

## 6. Current Status (Jan 2026)

- **V4 Stable**: The unified `flash_dual_ascent` is the production kernel.
- **Obsolescence**: `dual_ascent_implicit.cu` (V1) and `v2` have been removed.
- **Scale**: Verified up to $N=8192$ (Batch=1) and Batch=64 (N=1024).

For deep dive into implementation details, refer to `walkthrough.md`.
