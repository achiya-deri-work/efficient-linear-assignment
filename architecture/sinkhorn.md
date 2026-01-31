# Sinkhorn-Knopp Algorithm Architecture

## Overview

The Sinkhorn solver approximates the Optimal Transport plan (assignment) by iteratively normalizing rows and columns of the exponential cost matrix. It is the **fastest** solver in our suite for batches of dense inputs, especially with the `torch_compiled` backend.

## 1. Mathematical Formulation

We implement the **Log-Stabilized Sinkhorn** algorithm to ensure numerical stability even with small $\epsilon$ (regularization).

**Core Update Rule:**
$f \leftarrow \log(\mu) - \text{LogSumExp}(M_\epsilon + g)$
$g \leftarrow \log(\nu) - \text{LogSumExp}(M_\epsilon + f)$

Where:

- $M_\epsilon = -C / \epsilon$
- $f, g$ are dual potentials (Row/Col scalings in log domain)
- Final Assignment: $P = \exp(M_\epsilon + f + g)$

## 2. Implementation Strategies

### A. Torch Backend & Compilation

We utilize the **Nested Compile Region** pattern to optimize the iterative nature of Sinkhorn.

- **Inner Step**: The row/column updates are fused into a single graph via `@torch.compiler.nested_compile_region`. This allows Inductor to generate highly efficient fused kernels for the LogSumExp + Subtraction operations.
- **Shape Specialization**: The dynamic dispatcher (`compiled.py`) creates a fresh kernel for each unique $(B, N, M)$ shape, preventing graph re-compilation overhead ("Inductor Cache Misses").
- **TF32**: Enabled globally to accelerate Matrix Multiplications and Reductions on Ampere+ GPUs.

### B. Memory Optimization

- **In-Place Operations**: Where valid, operations aim to reuse buffers.
- **Vectorization**: The CUDA/Cutlass backends typically process 4-8 elements per thread (using `float4` or `float8` types) to maximize memory bandwidth.

## 3. Performance Characteristics

- **Complexity**: $O(K \cdot N \cdot M)$ where $K$ is iterations.
- **Sweet Spot**: Excellent for $N=128$ to $N=4096$.
- **Bottleneck**: Memory Bandwidth (reading $M_\epsilon$). The compiled backend minimizes this by fusing the read into the reduction.

## 4. Stability

- **Epsilon**: Lower $\epsilon$ yields sharper assignments (closer to permutation) but risks numerical overflow if not for log-stabilization.
- **Warmup**: The benchmark ensures `torch.no_grad()` is consistent to prevent `GradMode` guard failures in `torch.compile`.
