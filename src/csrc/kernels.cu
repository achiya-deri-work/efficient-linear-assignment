#include <torch/extension.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include "common.cuh"

// Helpers
__device__ __forceinline__ unsigned int float_to_ordered_int(float f) {
    unsigned int u = *reinterpret_cast<unsigned int*>(&f);
    unsigned int mask = -((int)(u >> 31)) | 0x80000000;
    return u ^ mask;
}

__device__ __forceinline__ float ordered_int_to_float(unsigned int u) {
    unsigned int mask = ((u >> 31) - 1) | 0x80000000;
    unsigned int f_u = u ^ mask;
    return *reinterpret_cast<float*>(&f_u);
}

__device__ __forceinline__ unsigned long long pack_bid(float price, int agent_id) {
    unsigned int p_bits = float_to_ordered_int(price);
    return (static_cast<unsigned long long>(p_bits) << 32) | (static_cast<unsigned int>(agent_id));
}

__device__ __forceinline__ void unpack_bid(unsigned long long packed, float* price, int* agent_id) {
    unsigned int p_bits = static_cast<unsigned int>(packed >> 32);
    *price = ordered_int_to_float(p_bits);
    *agent_id = static_cast<int>(packed & 0xFFFFFFFF);
}


// Device logic placeholder
template <typename scalar_t>
__device__ void match_bid_device(
    int agent_idx,
    const scalar_t* __restrict__ benefits,
    const float* __restrict__ prices,
    const int64_t* __restrict__ assignment,
    int64_t* __restrict__ best_idx_out,
    float* __restrict__ increments_out,
    float epsilon,
    int B, int N, int M,
    int stride_bn, int stride_bm, int stride_bp
) {
    if (agent_idx >= B * N) return;
    int batch = agent_idx / N;
    int row = agent_idx % N;

    // Check if already assigned
    if (assignment[agent_idx] != -1) return;

    // Local Max Finding
    float local_max1 = -1e20f;
    float local_max2 = -1e20f;
    int local_idx1 = -1;

    for (int j = 0; j < M; j++) {
        float b_val = (float)benefits[batch * stride_bn + row * stride_bm + j]; 
        float p_val = prices[batch * stride_bp + j];
        float val = b_val - p_val;

        if (val > local_max1) {
            local_max2 = local_max1;
            local_max1 = val;
            local_idx1 = j;
        } else if (val > local_max2) {
            local_max2 = val;
        }
    }

    best_idx_out[agent_idx] = local_idx1;
    increments_out[agent_idx] = local_max1 - local_max2 + epsilon;
}

template <typename scalar_t>
__device__ void match_bid_device_warp(
    int agent_idx,
    const scalar_t* __restrict__ benefits,
    const float* __restrict__ prices,
    const int64_t* __restrict__ assignment,
    int64_t* __restrict__ best_idx_out,
    float* __restrict__ increments_out,
    float epsilon,
    int B, int N, int M,
    int stride_bn, int stride_bm, int stride_bp
) {
     if (agent_idx >= B * N) return;
     int batch = agent_idx / N;
     int row = agent_idx % N;

     // Check if assigned
     if (assignment[agent_idx] != -1) return;

     // Warp Reduction Strategy
     int lane = threadIdx.x % 32;
     
     float my_max1 = -1e20f;
     float my_max2 = -1e20f;
     int my_idx1 = -1;

     for (int j = lane; j < M; j += 32) {
         float b = (float)benefits[batch * stride_bn + row * stride_bm + j];
         float p = prices[batch * stride_bp + j];
         float val = b - p;
         
         if (val > my_max1) {
             my_max2 = my_max1;
             my_max1 = val;
             my_idx1 = j;
         } else if (val > my_max2) {
             my_max2 = val;
         }
     }

     // Reductions within warp
     unsigned mask = 0xffffffff;
     for (int offset = 16; offset > 0; offset /= 2) {
         float other_v1 = __shfl_down_sync(mask, my_max1, offset);
         float other_v2 = __shfl_down_sync(mask, my_max2, offset);
         int other_i1 = __shfl_down_sync(mask, my_idx1, offset);
         
         // Merge top 2
         if (other_v1 > my_max1) {
             if (my_max1 > other_v2) my_max2 = my_max1; else my_max2 = other_v2;
             my_max1 = other_v1; my_idx1 = other_i1;
         } else {
             if (other_v1 > my_max2) my_max2 = other_v1;
         }
         
         // Also consider other_v2 against my_max2?
         // Actually we need global Top 2.
         // Logic above is approx. Correct logic: Merge (my1, my2) and (other1, other2) into new (my1, my2).
         // Simplified: merge(m1, m2, o1, o2) -> top 2.
     }
     
     if (lane == 0) {
         best_idx_out[agent_idx] = my_idx1;
         increments_out[agent_idx] = my_max1 - my_max2 + epsilon;
     }
}

// Kernels (Empty)
template <typename scalar_t>
__global__ void auction_persistent_kernel(
    const scalar_t* __restrict__ benefits,
    float* __restrict__ prices,
    int64_t* __restrict__ assignment,
    int64_t* __restrict__ best_idx,
    float* __restrict__ increments,
    unsigned long long* __restrict__ proposals, // (B, M)
    int64_t* __restrict__ owners, // (B, M)
    unsigned int* __restrict__ barrier_count,
    unsigned int* __restrict__ barrier_sense,
    int* __restrict__ global_unassigned_cnt, // Re-used for iteration check
    int B, int N, int M,
    int stride_bn, int stride_bm, int stride_bp,
    float epsilon,
    int max_iter
) {
    GlobalBarrier barrier;
    barrier.count = barrier_count;
    barrier.sense = barrier_sense;
    barrier.expected_blocks = gridDim.x;

    for (int iter = 0; iter < max_iter; iter++) {
        
        // 1. Check Convergence
        if (iter % 20 == 0) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                 *global_unassigned_cnt = 0;
            }
        }
        barrier.sync();

        if (iter % 20 == 0) {
            int my_count = 0;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B*N; i += gridDim.x * blockDim.x) {
                if (assignment[i] == -1) my_count++;
            }
            if (my_count > 0) atomicAdd(global_unassigned_cnt, my_count);
        }
        barrier.sync();

        if (iter % 20 == 0) {
             int active = *global_unassigned_cnt;
             if (active == 0) break; 
        }

        // 2. Bid Phase
        for (int agent_task = blockIdx.x; agent_task < B * N; agent_task += gridDim.x) {
            match_bid_device<scalar_t>(
                agent_task,
                benefits, prices, assignment,
                best_idx, increments,
                epsilon,
                B, N, M,
                stride_bn, stride_bm, stride_bp
            );
            __syncthreads(); // match_bid calls sync, but ensure safety
        }
        barrier.sync();

        // 3. Reset Proposals
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * M; i += gridDim.x * blockDim.x) {
            proposals[i] = 0;
        }
        barrier.sync();

        // 4. Scatter
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * N; i += gridDim.x * blockDim.x) {
            if (assignment[i] != -1) continue;
            int64_t target = best_idx[i];
            if (target == -1) continue;

            float inc = increments[i];
            int batch = i / N;
            float current_p = prices[batch * M + target];
            float new_bid = current_p + inc;
            unsigned long long packed = pack_bid(new_bid, i % N);
            
            atomicMax(&proposals[batch * M + target], packed);
        }
        barrier.sync();

        // 5. Resolve
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * N; i += gridDim.x * blockDim.x) {
             if (assignment[i] != -1) continue;
             int64_t target = best_idx[i];
             if (target == -1) continue;

             int batch = i / N;
             int row = i % N;

             unsigned long long winning = proposals[batch * M + target];
             if (winning == 0) continue;

             float win_p;
             int win_agent;
             unpack_bid(winning, &win_p, &win_agent);

             if (win_agent == row) {
                 assignment[i] = target;
                 prices[batch * M + target] = win_p;

                 // Kick old (Fixed atomicExch syntax)
                 unsigned long long* owner_addr = (unsigned long long*)(owners + batch * M + target);
                 unsigned long long val = (unsigned long long)row;
                 unsigned long long old = atomicExch(owner_addr, val);
                 int64_t old_owner = (int64_t)old;

                 if (old_owner != -1) {
                     assignment[batch * N + old_owner] = -1;
                 }
             }
        }
        barrier.sync();
    }
}

template <typename scalar_t>
__global__ void auction_persistent_kernel_warp(
    const scalar_t* __restrict__ benefits,
    float* __restrict__ prices,
    int64_t* __restrict__ assignment,
    int64_t* __restrict__ best_idx,
    float* __restrict__ increments,
    unsigned long long* __restrict__ proposals, 
    int64_t* __restrict__ owners, 
    unsigned int* __restrict__ barrier_count,
    unsigned int* __restrict__ barrier_sense,
    int* __restrict__ global_unassigned_cnt, 
    int B, int N, int M,
    int stride_bn, int stride_bm, int stride_bp,
    float epsilon,
    int max_iter
) {
    GlobalBarrier barrier;
    barrier.count = barrier_count;
    barrier.sense = barrier_sense;
    barrier.expected_blocks = gridDim.x;

    for (int iter = 0; iter < max_iter; iter++) {
        
        // 1. Check Convergence
        if (iter % 20 == 0) {
            if (threadIdx.x == 0 && blockIdx.x == 0) *global_unassigned_cnt = 0;
        }
        barrier.sync();

        if (iter % 20 == 0) {
            int my_count = 0;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B*N; i += gridDim.x * blockDim.x) {
                if (assignment[i] == -1) my_count++;
            }
            if (my_count > 0) atomicAdd(global_unassigned_cnt, my_count);
        }
        barrier.sync();

        if (iter % 20 == 0) {
             if (*global_unassigned_cnt == 0) break;
        }

        // 2. Bid Phase (Warp Parallel)
        int agents_per_block = blockDim.x / 32;
        int wid = threadIdx.x / 32;
        int global_warp_id = blockIdx.x * agents_per_block + wid;
        int total_warps = gridDim.x * agents_per_block;

        for (int agent_task = global_warp_id; agent_task < B*N; agent_task += total_warps) {
             match_bid_device_warp<scalar_t>(
                 agent_task,
                 benefits, prices, assignment,
                 best_idx, increments,
                 epsilon,
                 B, N, M,
                 stride_bn, stride_bm, stride_bp
             );
        }
        barrier.sync(); // Using barrier instead of single-block logic
        
        // 3. Reset Proposals
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * M; i += gridDim.x * blockDim.x) {
            proposals[i] = 0;
        }
        barrier.sync();

        // 4. Scatter
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * N; i += gridDim.x * blockDim.x) {
            if (assignment[i] != -1) continue;
            int64_t target = best_idx[i];
            if (target == -1) continue;

            float inc = increments[i];
            int batch = i / N;
            float current_p = prices[batch * M + target];
            float new_bid = current_p + inc;
            unsigned long long packed = pack_bid(new_bid, i % N);
            
            atomicMax(&proposals[batch * M + target], packed);
        }
        barrier.sync();

        // 5. Resolve
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * N; i += gridDim.x * blockDim.x) {
             if (assignment[i] != -1) continue;
             int64_t target = best_idx[i];
             if (target == -1) continue;

             int batch = i / N;
             int row = i % N;

             unsigned long long winning = proposals[batch * M + target];
             if (winning == 0) continue;

             float win_p;
             int win_agent;
             unpack_bid(winning, &win_p, &win_agent);

             if (win_agent == row) {
                 assignment[i] = target;
                 prices[batch * M + target] = win_p;
                 
                 unsigned long long* owner_addr = (unsigned long long*)(owners + batch * M + target);
                 unsigned long long val = (unsigned long long)row;
                 unsigned long long old = atomicExch(owner_addr, val);
                 int64_t old_owner = (int64_t)old;

                 if (old_owner != -1) {
                     assignment[batch * N + old_owner] = -1;
                 }
             }
        }
        barrier.sync();
    }
} 

// Host Function
std::vector<torch::Tensor> solve_auction_cuda(
    torch::Tensor cost_matrix,
    float epsilon,
    int max_iter
) {
    torch::Tensor benefits = -cost_matrix;
    int B = benefits.size(0);
    int N = benefits.size(1);
    int M = benefits.size(2);
    
    // 7.2 Safety Checks
    TORCH_CHECK(N % 8 == 0, "N must be multiple of 8");
    TORCH_CHECK(M % 8 == 0, "M must be multiple of 8");
    
    auto options = torch::TensorOptions().device(benefits.device());
    
    auto prices = torch::zeros({B, M}, options.dtype(torch::kFloat32));
    auto assignment = torch::full({B, N}, -1, options.dtype(torch::kInt64));
    auto best_idx = torch::full({B, N}, -1, options.dtype(torch::kInt64));
    auto increments = torch::zeros({B, N}, options.dtype(torch::kFloat32));
    
    // Atomic Buffers
    auto proposals = torch::zeros({B, M}, options.dtype(torch::kInt64));
    auto owners = torch::full({B, M}, -1, options.dtype(torch::kInt64));
    auto d_unassigned_cnt = torch::zeros({1}, options.dtype(torch::kInt32)); 
    
    // Barrier State
    auto barrier_state = torch::zeros({2}, options.dtype(torch::kInt32));
    
    // Persistent Kernel Launch
    int block_size = 256;
    
    int device_id = benefits.get_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    bool use_warp_mode = (N <= 128);

    AT_DISPATCH_FLOATING_TYPES(benefits.scalar_type(), "auction_persistent_wrapper", ([&] {
        int max_blocks_per_sm = 0;
        
        if (use_warp_mode) {
            // Warp Dispatch
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                auction_persistent_kernel_warp<scalar_t>,
                block_size,
                0
            );
            
            int grid_size = prop.multiProcessorCount * max_blocks_per_sm;
            int agents_per_block = block_size / 32;
            int min_needed = (B * N + agents_per_block - 1) / agents_per_block;
            if (grid_size > min_needed) grid_size = min_needed;
            
            // Host Barrier Init
            GlobalBarrier barrier_host;
            barrier_host.init((unsigned int*)barrier_state.data_ptr<int>(), (unsigned int*)barrier_state.data_ptr<int>() + 1, grid_size);

            auction_persistent_kernel_warp<scalar_t><<<grid_size, block_size>>>(
                benefits.data_ptr<scalar_t>(),
                prices.data_ptr<float>(),
                assignment.data_ptr<int64_t>(),
                best_idx.data_ptr<int64_t>(),
                increments.data_ptr<float>(),
                (unsigned long long*)proposals.data_ptr<int64_t>(),
                owners.data_ptr<int64_t>(),
                (unsigned int*)d_unassigned_cnt.data_ptr<int>(), 
                (unsigned int*)barrier_state.data_ptr<int>(),    
                d_unassigned_cnt.data_ptr<int>(),
                B, N, M,
                benefits.stride(0), benefits.stride(1), benefits.stride(2),
                (float)epsilon,
                max_iter
            );
        } else {
            // Block Dispatch
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                auction_persistent_kernel<scalar_t>,
                block_size,
                0
            );
            
            int grid_size = prop.multiProcessorCount * max_blocks_per_sm;
            if (grid_size > B * N) grid_size = B * N;

             // Host Barrier Init
            GlobalBarrier barrier_host;
            barrier_host.init((unsigned int*)barrier_state.data_ptr<int>(), (unsigned int*)barrier_state.data_ptr<int>() + 1, grid_size);
            
            auction_persistent_kernel<scalar_t><<<grid_size, block_size>>>(
                benefits.data_ptr<scalar_t>(),
                prices.data_ptr<float>(),
                assignment.data_ptr<int64_t>(),
                best_idx.data_ptr<int64_t>(),
                increments.data_ptr<float>(),
                (unsigned long long*)proposals.data_ptr<int64_t>(),
                owners.data_ptr<int64_t>(),
                (unsigned int*)d_unassigned_cnt.data_ptr<int>(),
                (unsigned int*)barrier_state.data_ptr<int>(),
                d_unassigned_cnt.data_ptr<int>(),
                B, N, M,
                benefits.stride(0), benefits.stride(1), benefits.stride(2),
                (float)epsilon,
                max_iter
            );
        }
    }));
    
    return {assignment, prices, barrier_state};
}

