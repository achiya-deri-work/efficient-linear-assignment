# Architecture Overview

`efficient-linear-assignment` is a high-performance, differentiable solver for the Linear Assignment Problem (LAP) using the **Auction Algorithm**. It is designed to scale efficiently on GPUs, handling batched inputs with thousands of agents ($N$) and objects ($M$).

## 1. Core Algorithm: The Auction Method
The solver uses the iterative Auction Algorithm with $\epsilon$-scaling.
*   **Forward Pass**: Agents "bid" on objects that offer the highest net benefit ($Benefit - Price$). Objects are assigned to the highest bidder, and their prices increase. This repeats until all agents are assigned or convergence criteria are met.
*   **Backward Pass**: Gradients are estimated via Implicit Maximum Likelihood Estimation (IMLE), allowing integration into end-to-end training pipelines.

## 2. System Components

### Frontend (`api.py`)
*   **Entry Point**: `linear_assignment(cost_matrix)`.
*   **Responsibilities**:
    *   Input Validation: Enforces $N, M$ are multiples of 8.
    *   Batch Handling: Supports `(B, N, M)` or `(N, M)` inputs.
    *   Autograd Interface: Wraps the solver in `torch.autograd.Function`.

### Backends
The library implements three pluggable backends, selectable via `backend=...`:

1.  **Torch Backend (`backend='torch'`)**
    *   **Implementation**: Pure PyTorch operations (vectorized).
    *   **Use Case**: Debugging, CPU fallback, reference implementation.
    *   **Strategy**: Uses `torch.topk` for bids and `scatter_reduce_` for price updates.

2.  **Triton Backend (`backend='triton'`)**
    *   **Implementation**: Custom Triton kernels for the "Bidding Phase".
    *   **Performance**: High throughput for large matrices.
    *   **Kernel**: `auction_bid_kernel` computes top-2 benefits per agent in parallel using block-level reductions.

3.  **C++/CUDA Backend (`backend='cpp'`)**
    *   **Implementation**: Optimized CUDA `.cu` kernels interfaced via PyTorch C++ Extension.
    *   **Performance**: Maximal control, using CUB-style warp primitives for reduction.
    *   **Dispatch**: Supports dynamic dispatch for FP32 and FP16 types.

## 3. Mixed Precision Strategy (Optimization)
To maximize GPU throughput while maintaining numerical correctness, the library employs a split-precision strategy:
*   **Data Transfer (Float16)**: The input `cost_matrix` is expected/allowed to be `Float16` (Half). This halves the PCIe and Global Memory bandwidth required to load the problem data.
*   **Computation (Float32)**: Internally, all solvers cast inputs to `Float32` before performing arithmetic.
    *   **Why?**: The Auction Algorithm relies on small $\epsilon$ increments. FP16 lacks the mantissa precision to accumulate small price updates over thousands of iterations without stalling (vanishing updates).
    *   **Implementation**:
        *   **C++**: `match_bid_kernel` accepts `Half*` benefits but explicitly casts to `float` for local accumulators and operations.
        *   **Triton**: Kernels load with `.to(tl.float32)`.

## 4. Robustness & Conflict Resolution
A critical challenge in parallel auction algorithms is handling **Degenerate Cases** (e.g., flat or zero cost matrices), which cause infinite loops or invalid assignments due to contention.

**Our Solution: Strict Collision Resolution**
*   **Logic**: During the update phase, if multiple agents bid effectively the same price (within tolerance) for the same object, the solver performs a strict "Election":
    1.  Scatter Agent IDs to a temporary `winners` buffer at the target object index.
    2.  Read back the winner ID.
    3.  Only the Agent who "won" the write is marked as assigned. Others remain unassigned and must bid again.
*   **Result**: Guarantees valid permutations (1-to-1 mapping) even when cost surfaces are completely flat.

## 5. Current Bottlenecks (See Optimization Plan)
The current architecture follows a "Ping-Pong" pattern where the GPU computes bids, and the CPU (Python) manages the loop control and updates.
*   See `architecture_optimization_plan.md` for the roadmap to "Persistent Kernels" and "Atomic Updates" to move the full loop to the GPU.
