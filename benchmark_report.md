# Efficient Linear Assignment Benchmark Report

## Overview

This report summarizes the performance of various linear assignment algorithms and backends implemented in the `efficient-linear-assignment` library. The benchmarks cover a range of problem sizes, precisions (FP32, FP16, BF16), and execution backends including PyTorch, `torch.compile`, Triton, and custom CUDA/Cutlass kernels.

## Key Findings

### 1. Sinkhorn

- **Small Scale (N=128)**: CUDA backend is fastest (~0.20ms), closely followed by `torch_compiled` (~0.25ms). Triton is competitive (~0.46ms).
- **Large Scale (N=4096)**: `torch_compiled` significantly outperforms all others (~15.8ms vs ~34.7ms for Triton and ~43.2ms for CUDA). This highlights the efficiency of Torch Dynamo's fusion and optimization for memory-bandwidth bound operations like Sinkhorn.
- **Precision**: Lower precision (FP16/BF16) yields consistent speedups, with `torch_compiled` maintaining its lead.

### 2. Dual Ascent

- **Small Scale (N=128)**: CUDA backend is superior (~0.71ms). `torch_compiled` is efficient (~1.1ms).
- **Large Scale (N=4096)**: `torch_compiled` (~62ms) dominates, being 2.5x faster than Triton and 3x faster than CUDA.
- **Flash Dual Ascent**: The `cuda_flash_v3` backend shows excellent scaling, matching or beating standard CUDA implementations, especially for larger sizes (56ms for N=4096), proving the efficacy of the implicit cost matrix approach.

### 3. Auction

- **Performace**: The Flash Auction (Triton Implicit) backend demonstrates reasonable performance (12ms for N=128, 118ms for N=4096).
- **Stability**: The standard Auction algorithm shows robust performance across backends, with Cutlass providing a reliable baseline.

## Detailed Results Summary

| Algo                | Backend        | Size (N) | Precision | Time (ms) | VRAM (MB) |
| ------------------- | -------------- | -------- | --------- | --------- | --------- |
| **Sinkhorn**        | torch_compiled | 4096     | FP32      | 15.83     | 1040      |
| Sinkhorn            | triton         | 4096     | FP32      | 34.69     | 1296      |
| Sinkhorn            | cuda           | 4096     | FP32      | 43.23     | 1296      |
| **DualAscent**      | torch_compiled | 4096     | FP32      | 62.37     | -         |
| **FlashDualAscent** | cuda_flash_v3  | 4096     | FP32      | 56.16     | -         |

## Conclusion

- **`torch.compile`** is the recommended backend for standard dense implementations (Sinkhorn, Dual Ascent) due to superior latency and memory fusion, especially at large scales.
- **Flash Backends** (Implicit) are essential for very large problems where materializing the cost matrix is prohibitive, offering competitive latency with significantly reduced memory footprint.
- **Triton** serves as a strong, portable alternative, particularly for custom kernels where CUDA C++ maintenance overhead is high.
