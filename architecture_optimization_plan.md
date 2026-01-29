# Architecture Optimization Plan: Maximizing GPU Saturation

## Current Bottlenecks
The current implementation adheres to a "Ping-Pong" architecture:
1.  **GPU**: Compute Bids (Kernel)
2.  **CPU/Python**: Sync, Check Termination, Update Ops (Scatter/Gather), Resolve Collisions
3.  **Repeat**

**Why this fails to saturate:**
*   **Launch Latency**: For highly efficient algorithms like Auction, the actual compute (finding top-2) is often faster than the Python overhead of launching the next step. The GPU spends time "waiting" for the CPU to tell it what to do next.
*   **Memory Roundtrips**: The "Collision Resolution" logic relies on global memory writes and multiple passes (Scatter -> Gather -> Check -> Rescatter) to ensure correctness. This thrashes global memory bandwidth.
*   **Scalar Logic on CPU**: Checking `unassigned.sum() == 0` moves data to CPU, forcing a synchronization barrier.

## Proposed Improvements (Axes of Optimization)

### 1. The "Persistent Agent" (Kernel Fusion)
**Goal**: Move the entire `while` loop onto the GPU.
Instead of launching 1000 kernels, launch **one** "Persistent Kernel" that runs the entire auction until convergence.

*   **Design**:
    *   **Persistent Blocks**: Launch exactly enough blocks to fill the GPU SMs (Streaming Multiprocessors).
    *   **Device-Side Loop**: The kernel contains the `while(unassigned > 0)` loop.
    *   **Global Synchronization**: Agents optimize synchronization using a "Grid Barrier" or "Split-Arrive" pattern. Since classic `__syncthreads()` only syncs a block, we can use a **Global Atomic Barrier** to sync all agents between the "Bid" and "Update" phases without exiting the kernel.
*   **Gain**: Zero CPU launch overhead. 100% GPU occupancy for the duration of the solve.

### 2. Atomic-Based Conflict Resolution
**Goal**: Replace the memory-heavy "Election Buffer" with hardware Atomics.
Current Python Logic:
```python
winners = zeros(M)
winners[obj] = agent_id # Last writer wins
verify = winners[obj] == agent_id
```
**Refined GPU Architecture**:
*   Use `atomicMax` on `Prices` buffer directly.
*   **Packed 64-bit Atomics**: To track *who* updated the price, pack `[Price (FP32) | AgentID (INT32)]` into a single `uint64`. Use `atomicMax` (treating bit-pattern as comparable integer if prices are positive, or use standard CAS loop).
*   **Benefit**: Resolves conflicts in a single pass of memory operations. No need for temporary "winner" arrays or multi-step verification.

### 3. Shared Memory Price Caching
**Goal**: Saturate L1/Shared Memory Bandwidth.
*   **Observation**: All agents read the *same* `prices` vector.
*   **Strategy**:
    *   If $M$ is small (e.g., < 4096), load the entire `prices` vector into Shared Memory at the start of the block.
    *   All threads read from Shared Memory for the inner loop.
    *   **Update Phase**: Threads simplify update prices in Shared Memory first, then flush to Global Memory.

### 4. CUDA Graphs (Intermediate Step)
**Goal**: Eliminate Python overhead without writing complex persistent kernels.
*   If full fusion is too complex, wrap the existing multi-kernel loop (Bid -> Search -> Update) in a **CUDA Graph**.
*   **Challenge**: Dynamic termination (`break` if converged).
*   **Solution**: "Launch Bounded" Graph. Capture a graph that runs $K$ iterations (e.g., 10). Launch it repeatedly until a device-side flag says "done".

## Roadmap

1.  **Phase 1: Atomic Upgrade (High Impact, Med Effort)**
    *   Write a dedicated `update_prices_kernel` in Triton/CUDA.
    *   Move the collision resolution logic from Python to this kernel using Atomics.
    *   Result: Loop becomes `Launch(Bid) -> Launch(Update)`. No Python tensor ops.

2.  **Phase 2: CUDA Graphs Integration**
    *   Wrap the `Bid -> Update` pair in `torch.cuda.make_graphed_callables`.

3.  **Phase 3: The Persistent Kernel (Ultimate Saturation)**
    *   Re-write the solver as a single monolithic kernel with an internal loop.
