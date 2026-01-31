# Auction Algorithm Architecture

## Overview

The Auction Algorithm is an iterative method that mimics a real-life auction. Agents bid on objects, raising prices until an equilibrium (valid assignment) is reached. It provides **exact** (or near-exact bounded by $\epsilon$) integer solutions.

## 1. Mathematical Formulation

**Iterative Loop:**

1. **Bidding**: Each unassigned agent finds the object $j^*$ offering maximum value $v_{ij} - p_j$.
2. **Bid**: The agent bids $b_{ij} = p_j + (v_{ij} - p_j) - (v_{ij'} - p_{j'}) + \epsilon$.
3. **Assignment**: Objects are assigned to the highest bidder. Values/Prices update.

## 2. Implementation: Block-Parallel Architecture

Our implementation (`auction_cutlass.cu`) introduces a **Block-per-Agent** paradigm to saturate modern GPUs.

### A. Architecture

- **Mapping**: Each CUDA Thread Block is assigned to ONE Agent (row).
- **Parallel Search**: The threads in the block cooperate to find the Top-2 values in the row (Best and Second Best) needed for the bid calculation.
- **Memory Coalescing**: By having a whole block read a row, we ensure perfectly coalesced 128-bit memory reads (`float4`). This resolves the "stride" inefficiency of naive Thread-per-Agent kernels.

### B. Reduction Strategy

- **Warp Reduce**: Threads perform warp-level reductions to find local maximums.
- **Shared Memory**: Warp results are aggregated in Shared Memory.
- **Block Reduce**: Final reduction computes the global Top-2 for the agent.

### C. Backend & Robustness

- **Backend Selection**: The `cutlass` backend is the primary high-performance engine.
- **Degeneracy Handling**: Includes logic to deterministically resolve ties (lowest ID wins) to prevent bidding cycles on flat cost surfaces.
- **Atomic Updates**: Bid submission uses `atomicMax` (or CAS loops) to update object prices safely in parallel.

## 3. Performance

- **Throughput**: Extremely high for large $N$ (e.g., 4096) due to full bandwidth saturation.
- **Latency**: Can be higher than Sinkhorn due to the variable number of interaction loops required to converge (data dependent).
