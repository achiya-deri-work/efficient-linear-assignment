#include <cuda_runtime.h>
#include <torch/types.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ------------------------------------------------------------------
// Helper: Float <-> Ordered Int
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// Helper: Global Barrier
// ------------------------------------------------------------------
struct GlobalBarrier {
    volatile unsigned int* count;
    volatile unsigned int* sense;
    int expected_blocks;

    __device__ void sync() {
        // Only thread 0 of each block participates in the global barrier
        if (threadIdx.x == 0) {
            unsigned int my_sense = atomicAdd((unsigned int*)sense, 0); // Read current
            // Atomic Inc
            unsigned int old_count = atomicAdd((unsigned int*)count, 1);
            if (old_count == expected_blocks - 1) {
                // Last block resets count and flips sense
                atomicExch((unsigned int*)count, 0); 
                __threadfence(); // Ensure count reset visible
                atomicExch((unsigned int*)sense, 1 - my_sense);
            } else {
                // Wait for sense to flip
                while (atomicAdd((unsigned int*)sense, 0) == my_sense) {
                     __nanosleep(100); // Backoff
                }
            }
        }
        __syncthreads(); // Block sync so all threads wait for thread 0
    }
};

// ------------------------------------------------------------------
// Device Function: Match Bid (Single Agent)
// ------------------------------------------------------------------
// Processes ONE agent 'row' (global index in B*N space).
// Assumes called by a WHOLE BLOCK (e.g. 256 threads).
template <typename scalar_t>
__device__ void match_bid_device(
    int agent_idx,
    const scalar_t* __restrict__ benefits, // (B, N, M)
    const float* __restrict__ prices,      // (B, M)
    const int64_t* __restrict__ assignment, // (B, N)
    int64_t* __restrict__ best_idx_out,    // (B, N)
    float* __restrict__ increments_out,    // (B, N)
    float epsilon,
    int B, int N, int M,
    int stride_bn, int stride_bm, int stride_bp
) {
    // Check Assignment
    if (assignment[agent_idx] != -1) {
        if (threadIdx.x == 0) {
            best_idx_out[agent_idx] = -1;
            increments_out[agent_idx] = 0.0f;
        }
        return;
    }

    int batch = agent_idx / N;
    int row = agent_idx % N;

    // Pointers
    const scalar_t* row_benefits = benefits + batch * stride_bn + row * stride_bm;
    const float* row_prices = prices + batch * stride_bp;

    // Local Top 2
    float local_max1 = -1e20f;
    float local_max2 = -1e20f;
    int local_idx1 = -1;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    bool can_vectorize = (std::is_same<scalar_t, float>::value) && (M % 4 == 0) && (reinterpret_cast<uintptr_t>(row_benefits) % 16 == 0);

    if (can_vectorize) {
        const float4* ben_v = reinterpret_cast<const float4*>(row_benefits);
        const float4* pr_v = reinterpret_cast<const float4*>(row_prices);
        int M_vec = M / 4;
        
        for (int i = tid; i < M_vec; i += stride) {
            float4 b = ben_v[i];
            float4 p = pr_v[i];
            // Unroll
            float v0 = b.x - p.x;
            if (v0 > local_max1) { local_max2 = local_max1; local_max1 = v0; local_idx1 = i*4+0; } else if (v0 > local_max2) { local_max2 = v0; }
            float v1 = b.y - p.y;
             if (v1 > local_max1) { local_max2 = local_max1; local_max1 = v1; local_idx1 = i*4+1; } else if (v1 > local_max2) { local_max2 = v1; }
            float v2 = b.z - p.z;
             if (v2 > local_max1) { local_max2 = local_max1; local_max1 = v2; local_idx1 = i*4+2; } else if (v2 > local_max2) { local_max2 = v2; }
            float v3 = b.w - p.w;
             if (v3 > local_max1) { local_max2 = local_max1; local_max1 = v3; local_idx1 = i*4+3; } else if (v3 > local_max2) { local_max2 = v3; }
        }
    } else {
        for (int col = tid; col < M; col += stride) {
            float val = static_cast<float>(row_benefits[col]) - row_prices[col];
            if (val > local_max1) {
                local_max2 = local_max1;
                local_max1 = val;
                local_idx1 = col;
            } else if (val > local_max2) {
                local_max2 = val;
            }
        }
    }

    // Block Reduction
    unsigned mask = 0xffffffff;
    
    // Warp Reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        float o_v1 = __shfl_down_sync(mask, local_max1, offset);
        float o_v2 = __shfl_down_sync(mask, local_max2, offset);
        int o_i1 = __shfl_down_sync(mask, local_idx1, offset);
        
        if (o_v1 > local_max1) {
             if (local_max1 > o_v2) local_max2 = local_max1; else local_max2 = o_v2;
             local_max1 = o_v1; local_idx1 = o_i1;
        } else {
             if (o_v1 > local_max2) local_max2 = o_v1;
        }
    }

    // Shared Mem for Cross-Warp
    static __shared__ float s_v1[32];
    static __shared__ float s_v2[32];
    static __shared__ int s_i1[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_v1[warp_id] = local_max1;
        s_v2[warp_id] = local_max2;
        s_i1[warp_id] = local_idx1;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        if (lane_id < num_warps) {
            local_max1 = s_v1[lane_id];
            local_max2 = s_v2[lane_id];
            local_idx1 = s_i1[lane_id];
        } else {
            local_max1 = -1e20f; local_max2 = -1e20f; local_idx1 = -1;
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            float o_v1 = __shfl_down_sync(mask, local_max1, offset);
            float o_v2 = __shfl_down_sync(mask, local_max2, offset);
            int o_i1 = __shfl_down_sync(mask, local_idx1, offset);
            if (o_v1 > local_max1) {
                 if (local_max1 > o_v2) local_max2 = local_max1; else local_max2 = o_v2;
                 local_max1 = o_v1; local_idx1 = o_i1;
            } else { if (o_v1 > local_max2) local_max2 = o_v1; }
        }

        if (lane_id == 0) {
            best_idx_out[agent_idx] = local_idx1;
            increments_out[agent_idx] = local_max1 - local_max2 + epsilon;
        }
    }
}

// ------------------------------------------------------------------
// Persistent Solver Kernel
// ------------------------------------------------------------------
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
    // Persistent Grid Logic
    // Grid: Fixed number of blocks.
    // We loop tasks over the grid.
    
    // Setup Barrier
    GlobalBarrier barrier;
    barrier.count = barrier_count;
    barrier.sense = barrier_sense;
    barrier.expected_blocks = gridDim.x;

    for (int iter = 0; iter < max_iter; iter++) {
        
        // ---------------------------------------------------------
        // 1. Check Convergence (Start of loop or skip first?)
        // Let's check at start. If count == 0, break.
        // Requires global sum. 
        // Optimization: Check every K iters.
        // Initialize: global_unassigned_cnt = B*N at start of kernel (by host).
        
        if (iter % 20 == 0) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                 // Leader zero outs counter for re-calculation
                 // Actually relying on atomicSub from previous Resolve is tricky if counts drift.
                 // Safer: Sum 'assignment == -1' in parallel.
                 *global_unassigned_cnt = 0;
            }
        }
        barrier.sync();

        if (iter % 20 == 0) {
            // Count unassigned
            int my_count = 0;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B*N; i += gridDim.x * blockDim.x) {
                if (assignment[i] == -1) my_count++;
            }
            // Atomic Add to global
            if (my_count > 0) atomicAdd(global_unassigned_cnt, my_count);
        }
        barrier.sync();

        if (iter % 20 == 0) {
             int active = *global_unassigned_cnt;
             if (active == 0) break; // Converged
        }

        // ---------------------------------------------------------
        // 2. Bid Phase
        // Each BLOCK picks an agent and runs match_bid_device
        // Map: blockIdx.x -> Agent Index?
        // gridDim.x blocks. B*N agents.
        // If we map 1 Agent per Block, we need loop:
        for (int agent_task = blockIdx.x; agent_task < B * N; agent_task += gridDim.x) {
            match_bid_device<scalar_t>(
                agent_task,
                benefits, prices, assignment,
                best_idx, increments,
                epsilon,
                B, N, M,
                stride_bn, stride_bm, stride_bp
            );
            __syncthreads(); // match_bid calls sync, but ensure safety within loop if needed
        }
        barrier.sync();

        // ---------------------------------------------------------
        // 3. Reset Proposals
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * M; i += gridDim.x * blockDim.x) {
            proposals[i] = 0;
        }
        barrier.sync();

        // ---------------------------------------------------------
        // 4. Scatter
        // 1D Thread Stride
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * N; i += gridDim.x * blockDim.x) {
            // Check active
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

        // ---------------------------------------------------------
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

                 // Kick old
                 int64_t old_owner = atomicExch((unsigned long long*)&owners[batch * M + target], (unsigned long long)row);
                 if (old_owner != -1) {
                     assignment[batch * N + old_owner] = -1;
                 }
             }
        }
        barrier.sync();
    }
}


// ------------------------------------------------------------------
// Legacy Wrappers (Still needed for non-persistent or debug)
// ------------------------------------------------------------------

// Simple Kernel wrappers re-using device functions where possible or kept as is
// if structure is wildly different. The persistent kernel covers the logic.
// But we keep the original kernels slightly modified or use persistent kernel for everything?
// The user might want the option. Let's keep original kernels but refactor them to use the device function if valid.
// Actually original 'match_bid_kernel' was 1 block per agent.
// We can make match_bid_kernel just call the device fn.

template <typename scalar_t>
__global__ void match_bid_kernel_wrapper(
    const scalar_t* __restrict__ benefits,
    const float* __restrict__ prices,
    const int64_t* __restrict__ assignment,
    int64_t* __restrict__ best_idx_out,
    float* __restrict__ increments_out,
    float epsilon,
    int B, int N, int M,
    int stride_bn, int stride_bm, int stride_bp
) {
    // Grid (N, B) -> blockIdx.x = row, blockIdx.y = batch
    int row = blockIdx.x;
    int batch = blockIdx.y;
    int agent_idx = batch * N + row;
    if (batch >= B || row >= N) return;

    match_bid_device<scalar_t>(
        agent_idx, benefits, prices, assignment, best_idx_out, increments_out,
        epsilon, B, N, M, stride_bn, stride_bm, stride_bp
    );
}

// Keep scatter/resolve separate kernels for the iterative host-side solver
__global__ void scatter_kernel_wrapper(
     const int64_t* __restrict__ best_idx,
     const float* __restrict__ increments,
     const float* __restrict__ prices,
     const int64_t* __restrict__ assignment,
     unsigned long long* __restrict__ proposals,
     int B, int N, int M
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= B * N) return;
    
    int batch = global_id / N;
    int row = global_id % N;
    
    if (assignment[global_id] != -1) return;
    int64_t target = best_idx[global_id];
    if (target == -1) return;
    
    float inc = increments[global_id];
    float current_p = prices[batch * M + target];
    float new_bid = current_p + inc;
    unsigned long long packed = pack_bid(new_bid, row);
    
    atomicMax(&proposals[batch * M + target], packed);
}

__global__ void resolve_kernel_wrapper(
    const int64_t* __restrict__ best_idx,
    int64_t* __restrict__ assignment,
    float* __restrict__ prices,
    const unsigned long long* __restrict__ proposals,
    int64_t* __restrict__ owners, // (B, M)
    int* __restrict__ unassigned_count, 
    int B, int N, int M
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= B * N) return;
    int batch = global_id / N;
    int row = global_id % N; 
    
    if (assignment[global_id] != -1) return;
    int64_t target = best_idx[global_id];
    if (target == -1) return;
    
    unsigned long long winning = proposals[batch * M + target];
    if (winning == 0) return; 
    
    float win_p;
    int win_agent;
    unpack_bid(winning, &win_p, &win_agent);
    
    if (win_agent == row) {
        assignment[global_id] = target;
        prices[batch * M + target] = win_p;
        
        int64_t old_owner = atomicExch((unsigned long long*)(owners + batch * M + target), (unsigned long long)row);
        if (old_owner != -1) {
            assignment[batch * N + old_owner] = -1;
            // No unassigned_count update here needed for simple check
        }
    }
}

// Reset wrapper
__global__ void reset_proposals_wrapper(unsigned long long* proposals, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) proposals[idx] = 0;
}


// ------------------------------------------------------------------
// Host Interface
// ------------------------------------------------------------------

void launch_bid_kernel_cuda(
    torch::Tensor benefits,
    torch::Tensor prices,
    torch::Tensor assignment,
    torch::Tensor best_idx,
    torch::Tensor increments,
    double epsilon
) {
    int B = benefits.size(0);
    int N = benefits.size(1);
    int M = benefits.size(2);
    dim3 grid(N, B);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(benefits.scalar_type(), "match_bid_kernel", ([&] {
        match_bid_kernel_wrapper<scalar_t><<<grid, 256>>>(
            benefits.data_ptr<scalar_t>(),
            prices.data_ptr<float>(),     
            assignment.data_ptr<int64_t>(),
            best_idx.data_ptr<int64_t>(),
            increments.data_ptr<float>(), 
            (float)epsilon,
            B, N, M,
            benefits.stride(0), benefits.stride(1),
            prices.stride(0)
        );
    }));
}


// Unified Entry Point
std::vector<torch::Tensor> solve_auction_cuda(
    torch::Tensor cost_matrix,
    double epsilon,
    int max_iter,
    bool persistent_mode // New Flag
) {
    torch::Tensor benefits = -cost_matrix;
    int B = benefits.size(0);
    int N = benefits.size(1);
    int M = benefits.size(2);
    auto options = torch::TensorOptions().device(benefits.device());
    
    auto prices = torch::zeros({B, M}, options.dtype(torch::kFloat32));
    auto assignment = torch::full({B, N}, -1, options.dtype(torch::kInt64));
    auto best_idx = torch::full({B, N}, -1, options.dtype(torch::kInt64));
    auto increments = torch::zeros({B, N}, options.dtype(torch::kFloat32));
    
    // Atomic Buffers
    auto proposals = torch::zeros({B, M}, options.dtype(torch::kInt64));
    auto owners = torch::full({B, M}, -1, options.dtype(torch::kInt64));
    auto d_unassigned_cnt = torch::zeros({1}, options.dtype(torch::kInt32)); // Reused for barrier/counter

    // Barrier State (Keep alive)
    auto barrier_state = torch::empty({0}, options.dtype(torch::kInt32)); 

    if (persistent_mode) {
        // Persistent Kernel Launch
        int block_size = 256;
        
        int device_id = benefits.get_device();
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        int num_sms = prop.multiProcessorCount;
        
        // Init Barrier
        barrier_state = torch::zeros({2}, options.dtype(torch::kInt32));
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(benefits.scalar_type(), "auction_persistent_wrapper", ([&] {
            int max_blocks_per_sm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                auction_persistent_kernel<scalar_t>,
                block_size,
                0
            );
            
            // Fallback to Single Block Serial Execution to prevent deadlocks in GlobalBarrier
            int grid_size = 1;
            
            // Optimization & Safety
            if (grid_size < 1) grid_size = 1;

            int tasks = B * N;
            int needed = (tasks + block_size - 1) / block_size;
            if (grid_size > needed) grid_size = needed;
            if (grid_size < 1) grid_size = 1;
            
            auction_persistent_kernel<scalar_t><<<grid_size, block_size>>>(
                benefits.data_ptr<scalar_t>(),
                prices.data_ptr<float>(),
                assignment.data_ptr<int64_t>(),
                best_idx.data_ptr<int64_t>(),
                increments.data_ptr<float>(),
                reinterpret_cast<unsigned long long*>(proposals.data_ptr<int64_t>()),
                owners.data_ptr<int64_t>(),
                (unsigned int*)barrier_state.data_ptr<int32_t>(),     // Count
                (unsigned int*)barrier_state.data_ptr<int32_t>() + 1, // Sense
                d_unassigned_cnt.data_ptr<int32_t>(),
                B, N, M,
                benefits.stride(0), benefits.stride(1), prices.stride(0),
                (float)epsilon,
                max_iter
            );
        }));
        
    } else {
        // Legacy Host Loop
        int check_interval = 20;
        int total_agents = B * N;
        int total_objs = B * M;
        int update_grid = (total_agents + 256 - 1) / 256;
        int prop_grid = (total_objs + 256 - 1) / 256;
        dim3 bid_grid(N, B);

        for(int i = 0; i < max_iter; i++) {
             bool do_check = (i % check_interval == 0) || (i == max_iter - 1);
        
             // Bid
             AT_DISPATCH_FLOATING_TYPES_AND_HALF(benefits.scalar_type(), "match_bid", ([&] {
                 match_bid_kernel_wrapper<scalar_t><<<bid_grid, 256>>>(
                     benefits.data_ptr<scalar_t>(),
                     prices.data_ptr<float>(),
                     assignment.data_ptr<int64_t>(),
                     best_idx.data_ptr<int64_t>(),
                     increments.data_ptr<float>(),
                     (float)epsilon,
                     B, N, M,
                     benefits.stride(0), benefits.stride(1), prices.stride(0)
                 );
             }));
             
             // Reset
             reset_proposals_wrapper<<<prop_grid, 256>>>(
                 reinterpret_cast<unsigned long long*>(proposals.data_ptr<int64_t>()), 
                 total_objs
             );
             
             // Scatter
             scatter_kernel_wrapper<<<update_grid, 256>>>(
                best_idx.data_ptr<int64_t>(),
                increments.data_ptr<float>(),
                prices.data_ptr<float>(),
                assignment.data_ptr<int64_t>(),
                reinterpret_cast<unsigned long long*>(proposals.data_ptr<int64_t>()),
                B, N, M
             );
             
             // Resolve
             resolve_kernel_wrapper<<<update_grid, 256>>>(
                best_idx.data_ptr<int64_t>(),
                assignment.data_ptr<int64_t>(),
                prices.data_ptr<float>(),
                reinterpret_cast<const unsigned long long*>(proposals.data_ptr<int64_t>()),
                owners.data_ptr<int64_t>(),
                nullptr,
                B, N, M
             );
             
             if (do_check) {
                 int cnt = (assignment.view(-1) == -1).sum().item<int>();
                 if (cnt == 0) break;
             }
        }
    }
    
    return {assignment, prices, barrier_state};
}
