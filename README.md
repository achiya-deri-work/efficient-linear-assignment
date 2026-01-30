# Efficient Differentiable Linear Assignment Suite

A library of high-performance, differentiable solvers for **Optimal Transport**, **Linear Assignment**, and **Routing** problems, accelerated by **Triton** and **CUDA**.

This suite includes:

1.  **Auction Algorithm**: Combinatorial exact matching (LAP).
2.  **Log-Stabilized Sinkhorn**: Differentiable Entropic Optimal Transport.
3.  **L2-Regularized Dual Ascent**: Sparse, structured attention/matching.
4.  **MaxScore Routing**: Capacity-constrained routing for MoE/Expert models.

Designed for deep learning pipelines where assignments must be computed dynamically and differentiably during training.

## Features

- **🚀 High Performance**: Faster than `scipy.optimize.linear_sum_assignment` and purely solving on GPU.
- **🔥 Differentiable**: Supports backward passes via Implicit Maximum Likelihood Estimation (IMLE).
- **⚡ Mixed Precision**: Optimizes memory bandwidth by accepting `Float16` inputs while maintaining `Float32` precision for internal stability.
- **🛡️ Robust**: Handles degenerate cases (all-zero costs, flat surfaces) with strict collision resolution guarantees.
- **🔌 Pluggable Backends**: Switch between `torch` (reference), `triton` (fused), and `cpp` (CUDA optimized) implementations.

## Installation

```bash
pip install .
```

_Requirements_:

- PyTorch >= 2.0
- Triton (for `backend='triton'`)
- CUDA Toolkit (for `backend='cpp'`)

## Usage

### 1. Linear Assignment (Exact - Auction)

Ideal for exact bipartite matching (1-to-1).

```python
from efficient_linear_assignment import linear_assignment

# B=Batch, N=Workers, M=Jobs (N=M for LAP)
cost_matrix = torch.rand((4, 128, 128), device='cuda')

# Returns Indices (B, N)
indices = linear_assignment(
    cost_matrix,
    backend='cpp', # 'torch', 'triton', 'cpp'
    epsilon=1e-2
)
```

### 2. Entropic Optimal Transport (Sinkhorn)

Differentiable soft-matching with entropic regularization.

```python
from efficient_linear_assignment import log_stabilized_sinkhorn

# Returns Transport Matrix P (B, N, M)
P = log_stabilized_sinkhorn(
    cost_matrix,
    epsilon=0.1,
    num_iters=20,
    backend='triton' # 'torch', 'triton', 'cuda'
)
# P sums to 1/N rows and 1/M cols (uniform marginals by default)
```

### 3. Sparse Matching (Dual Ascent)

L2-Regularized OT producing sparse, structured transport plans.

```python
from efficient_linear_assignment import l2_regularized_dual_ascent

# Returns Transport Matrix P (B, N, M)
P = l2_regularized_dual_ascent(
    cost_matrix,
    epsilon=1.0,
    num_iters=10,
    backend='cuda'
)
# Result is strictly sparse (ReLU activated)
```

### 4. Expert Routing (MaxScore)

Capacity-constrained routing for MoE (Mixture of Experts). Ensures experts are balanced.

```python
from efficient_linear_assignment import max_score_routing

# logits: (B, Tokens, Experts)
logits = torch.randn(1, 1024, 8, device='cuda')

# Returns Normalized Probabilities (B, Tokens, Experts)
scores = max_score_routing(
    logits,
    capacity_factor=1.0,
    backend='triton'
)
```

### Input Constraints & Padded Inputs

To maximize kernel efficiency, input dimensions **$N$ and $M$ must be multiples of 8**.
If your data size is arbitrary, please pad your input tensors before calling the solver:

```python
from efficient_linear_assignment.utils import pad_input, check_dims

cost_matrix = ... # shape (1, 10, 10)
# Pad to (1, 16, 16)
padded_cost, original_shape = pad_input(cost_matrix, multiple=8)

# Solve
indices = linear_assignment(padded_cost)

# Crop back to valid range logic...
```

## Architecture

This library uses a "Ping-Pong" architecture where:

1.  **GPU Kernel**: Computes bids (Top-2 benefits) for all agents in parallel.
2.  **Python Native**: Performs price updates and collision resolution on the GPU tensors.

For a deep dive into the design and future optimization plans (Persistent Kernels, Atomics), please see:

- [Architecture Overview](architecture.md)
- [Optimization Roadmap](architecture_optimization_plan.md)

## Performance

Results from benchmarking on NVIDIA GeForce RTX 5070 Ti (B=1).

**Comparison vs. Baseline (`torch`) and External (`torch-linear-assignment` Hungarian implementation).**

**Comparison vs. Baseline (`torch`) at 4096x4096**

| Algo            | Precision | Torch (ms) | Triton (ms) | **CUDA (ms)** |
| :-------------- | :-------- | :--------- | :---------- | :------------ |
| **Sinkhorn**    | FP32      | 24.97      | 8.22        | **10.33**     |
|                 | FP16      | 10.09      | 7.15        | **8.92**      |
|                 | **BF16**  | 10.17      | 7.30        | **8.70**      |
| **Dual Ascent** | FP32      | 18.91      | 4.39        | **5.27**      |
|                 | FP16      | 17.90      | 3.63        | **4.43**      |
|                 | **BF16**  | 17.91      | 3.68        | **4.31**      |
| **Routing**     | FP32      | 28.44      | 6.63        | **7.97**      |
|                 | FP16      | 27.06      | 5.53        | **6.45**      |
|                 | **BF16**  | 27.05      | 5.48        | **6.56**      |

_(Benchmarks on NVIDIA GPU, B=1, N=4096)_

### Benchmark Visuals

**1. Execution Time (Log Scale)**
![Log Time](assets/benchmark_time_log.png)

**2. GPU Backends Comparison (Linear Scale)**
![Linear Time](assets/benchmark_time_linear.png)

**3. Speedup Factor vs Torch**
![Speedup](assets/benchmark_speedup.png)

## Backends

| Backend      | Description                | Pros                                          | Cons                                 |
| :----------- | :------------------------- | :-------------------------------------------- | :----------------------------------- |
| **`torch`**  | Pure PyTorch operations    | No extra deps, stable reference.              | Slower, high overhead.               |
| **`triton`** | OpenAI Triton Kernels      | Fast, readable.                               | **~5x Slower than C++** for small N. |
| **`cpp`**    | **Proprietary C++ / CUDA** | **Fastest (<6ms)**. Async Persistent Kernels. | Requires build step.                 |

## Contributing

We are actively looking for contributors! If you're interested in high-performance GPU kernels, optimization, or differentiable algorithms, please feel free to open a PR or issue.

Current areas of interest:

- **BF16 Support for CUDA:** Implementing BFloat16 dispatch for CUDA kernels.
- **Kernel Optimization:** Further tuning of tile sizes and memory access patterns.
- **New Algorithms:** Implementing additional optimal transport or assignment algorithms.

## License

[License Name]
