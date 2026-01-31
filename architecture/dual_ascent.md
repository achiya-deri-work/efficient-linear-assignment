# Dual Ascent Algorithm Architecture

## Overview

The Dual Ascent solver tackles the Linear Assignment Problem by optimizing the dual variables (potentials) via Newton-Coordinate Descent. Unlike Sinkhorn (entropic regularization), Dual Ascent is typically used with **L2 Regularization**, leading to sparse transport plans.

## 1. Mathematical Formulation

We solve the dual of the L2-regularized OT problem. The updates are derived from finding the root of the gradient (marginal constraint error) for each row/column sequentially.

**Update Rule (Row-wise):**

1. Compute Slack: $T = \alpha + \beta - C$
2. Active Set: $S = \{j \mid T_{ij} > 0\}$
3. Gradient: $\nabla = \sum_{j \in S} 1 / \epsilon$
4. Residual: $R = \mu_i - \sum_{j \in S} T_{ij}/\epsilon$
5. Update: $\alpha_i \leftarrow \alpha_i + R / \nabla$

This is repeated for columns ($\beta$).

## 2. Implementation Strategies

### A. Torch Compiled (Nested)

Similar to Sinkhorn, we leverage `torch.compile` with the **Nested Compile Region** pattern.

- **Sparse Reductions**: The algorithm involves masking operations (`active_mask = T > 0`) followed by reductions. Compiling this allows kernel fusion, avoiding multiple reads/writes of the large $N \times M$ matrix.
- **Fresh Dispatch**: Shape-specialized kernels via `compiled.py`.

### B. CUDA / CUTLASS

- **Parallelism**: Efficiently parallelizes the reduction over rows/cols.
- **Vectorization**: Uses vector loads (`float4`) to read inputs, significantly reducing memory bandwidth pressure compared to scalar kernels.

## 3. Comparison to Sinkhorn

- **Sparsity**: Dual Ascent produces sparse outputs (many zeros), whereas Sinkhorn produces dense probability maps.
- **Convergence**: Often converges in fewer iterations (10-20) for a "good enough" assignment.
- **Performance**: Generally faster per iteration than Sinkhorn due to simpler arithmetic (ReLU instead of Exp/Log), provided the sparsity is leveraged or fused.

## 4. Key Optimizations

- **TF32**: Enabled for accelerated accumulation.
- **Active Set Caching**: (Conceptually) The active set changes slowly; optimized kernels can exploit this, though currently we recompute correctness at each step for robustness.
