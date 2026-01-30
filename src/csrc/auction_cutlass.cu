#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/fast_math.h>

#include "common.cuh"

// ----------------------------------------------------------------------
// CUTLASS Auction Kernel (Exact)
// ----------------------------------------------------------------------

// Atomic Helper
__device__ void update_bid_atomic(unsigned long long* addr, float bid_val, int bidder_id) {
    unsigned int f_bits = __float_as_uint(bid_val);
    unsigned long long new_packed = (static_cast<unsigned long long>(f_bits) << 32) | static_cast<unsigned int>(bidder_id);
    
    unsigned long long old = *addr;
    // Assume sorted as unsigned int holds for positive floats
    // If floats can be negative, this simple logic breaks.
    // We assume cost logic produces positive prices or we perform monotonic shifts.
    while (new_packed > old) {
        unsigned long long ass = atomicCAS(addr, old, new_packed);
        if (ass == old) break;
        old = ass;
    }
}

// Helper: Pair reduction for (val, idx)
__device__ void update_top2(float val, int idx, float& m1, int& i1, float& m2) {
    if (val > m1) {
        m2 = m1;
        m1 = val;
        i1 = idx;
    } else if (val > m2) {
        m2 = val;
    }
}

template <typename T, int VecSize>
__global__ void auction_bid_kernel_block(
    const T* __restrict__ C_ptr,
    const T* __restrict__ prices_ptr,
    const int* __restrict__ assignment_ptr,
    unsigned long long* __restrict__ bids_packed,
    int B, int N, int M,
    float epsilon
) {
    // One Block per Agent.
    int agent_idx = blockIdx.x;
    if (agent_idx >= B * N) return;
    
    // Check assignment
    // (Optimization: can we skip block entirely? Yes)
    if (assignment_ptr[agent_idx] != -1) return;
    
    int tid = threadIdx.x;
    
    int b = agent_idx / N;
    int i = agent_idx % N;
    
    const T* row_C = C_ptr + (b * N * M) + (i * M);
    const T* vec_prices = prices_ptr + (b * M);
    
    float l_m1 = -1e30f;
    float l_m2 = -1e30f;
    int l_i1 = -1;
    
    using VecT = cutlass::Array<T, VecSize>;
    int limit = (M / VecSize) * VecSize;
    int stride = blockDim.x * VecSize;
    
    for (int j = tid * VecSize; j < limit; j += stride) {
        VecT c_val = *reinterpret_cast<const VecT*>(row_C + j);
        VecT p_val = *reinterpret_cast<const VecT*>(vec_prices + j);
        
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < VecSize; ++k) {
            float val = -static_cast<float>(c_val[k]) - static_cast<float>(p_val[k]);
            update_top2(val, j+k, l_m1, l_i1, l_m2);
        }
    }
    // Tail
    for (int j = limit + tid; j < M; j += blockDim.x) {
        float val = -static_cast<float>(row_C[j]) - static_cast<float>(vec_prices[j]);
        update_top2(val, j, l_m1, l_i1, l_m2);
    }
    
    // Block Reduction
    // 1. Warp Reduce
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other_m1 = __shfl_down_sync(0xFFFFFFFF, l_m1, offset);
        int other_i1 = __shfl_down_sync(0xFFFFFFFF, l_i1, offset);
        float other_m2 = __shfl_down_sync(0xFFFFFFFF, l_m2, offset);
        
        // Merge (me, other)
        if (other_m1 > l_m1) {
             // other is best.
             // merge my m1 into m2 candidate
             float cand_m2 = (l_m1 > other_m2) ? l_m1 : other_m2;
             l_m1 = other_m1;
             l_i1 = other_i1;
             l_m2 = (cand_m2 > l_m2) ? cand_m2 : l_m2;
        } else {
             // i am best.
             // merge other m1 into m2 candidate
             float cand_m2 = (other_m1 > l_m2) ? other_m1 : l_m2;
             l_m2 = (cand_m2 > other_m2) ? cand_m2 : other_m2; // Wait, max(other_m1, l_m2, other_m2)?
             // logic: new_m2 = max(l_m2, other_m2, min(l_m1, other_m1))
             // since l_m1 >= other_m1, new_m2 = max(l_m2, other_m2, other_m1).
             l_m2 = (other_m1 > l_m2) ? other_m1 : l_m2;
             l_m2 = (other_m2 > l_m2) ? other_m2 : l_m2;
        }
    }
    
    // 2. Shared Memory Reduce (across warps)
    __shared__ float sm_m1[32];
    __shared__ int sm_i1[32];
    __shared__ float sm_m2[32];
    
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    
    if (laneId == 0) {
        sm_m1[warpId] = l_m1;
        sm_i1[warpId] = l_i1;
        sm_m2[warpId] = l_m2;
    }
    __syncthreads();
    
    // First warp reduces the shared results
    if (warpId == 0) {
        int num_warps = blockDim.x / warpSize;
        l_m1 = (tid < num_warps) ? sm_m1[laneId] : -1e30f;
        l_i1 = (tid < num_warps) ? sm_i1[laneId] : -1;
        l_m2 = (tid < num_warps) ? sm_m2[laneId] : -1e30f;
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            float other_m1 = __shfl_down_sync(0xFFFFFFFF, l_m1, offset);
            int other_i1 = __shfl_down_sync(0xFFFFFFFF, l_i1, offset);
            float other_m2 = __shfl_down_sync(0xFFFFFFFF, l_m2, offset);
            
            if (other_m1 > l_m1) {
                 float cand_m2 = (l_m1 > other_m2) ? l_m1 : other_m2;
                 l_m1 = other_m1;
                 l_i1 = other_i1;
                 l_m2 = (cand_m2 > l_m2) ? cand_m2 : l_m2;
            } else {
                 float cand_m2 = (other_m1 > l_m2) ? other_m1 : l_m2;
                 l_m2 = (cand_m2 > other_m2) ? cand_m2 : l_m2; // fix logic
                 l_m2 = (other_m2 > l_m2) ? other_m2 : l_m2;
            }
        }
        
        if (tid == 0 && l_i1 != -1) {
            float bid_inc = l_m1 - l_m2 + epsilon;
            float current_p = static_cast<float>(prices_ptr[b * M + l_i1]); // Wait, need generic pointer logic if M large?
            // vec_prices is correct base.
            // But we need to read 'price' at 'l_i1'.
            // Accessing prices[l_i1] is safe.
            float bid_val = static_cast<float>(prices_ptr[(long long)b * M + l_i1]) + bid_inc;
            update_bid_atomic(bids_packed + ((long long)b * M + l_i1), bid_val, i);
        }
    }
}

template <typename T>
__global__ void auction_assign_kernel(
    T* __restrict__ prices_ptr,
    int* __restrict__ assignment_ptr,
    int* __restrict__ owner_ptr,
    unsigned long long* __restrict__ bids_packed,
    int* __restrict__ unassigned_count_ptr,
    int B, int N, int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * M) return;
    
    int b = idx / M;
    int j = idx % M;
    
    unsigned long long packet = bids_packed[idx];
    
    // Reset bid scratch for next iter
    bids_packed[idx] = 0; 
    
    if (packet == 0) return; // No bid
    
    int bidder_i = (int)(packet & 0xFFFFFFFF);
    float bid_val = __uint_as_float((unsigned int)(packet >> 32));
    
    // Update Price
    prices_ptr[idx] = static_cast<T>(bid_val);
    
    // Assign Item j to bidder_i
    int old_owner = owner_ptr[idx];
    
    if (old_owner != -1) {
        // Kick out old owner
        assignment_ptr[b * N + old_owner] = -1;
        // Atomic add to unassigned? Or just let loop handle it?
        // We need to count unassigned to know when to stop.
        atomicAdd(unassigned_count_ptr, 1);
    }
    
    // Set new owner
    owner_ptr[idx] = bidder_i;
    assignment_ptr[b * N + bidder_i] = j; // Person i gets j
    atomicAdd(unassigned_count_ptr, -1);
}

std::vector<torch::Tensor> auction_cutlass_forward(
    torch::Tensor C,
    float epsilon,
    int max_iter
) {
    auto B = C.size(0);
    auto N = C.size(1);
    auto M = C.size(2);
    auto opts = C.options();
    
    // State Tensors
    auto prices = torch::zeros({B, M}, opts);
    auto assignment = torch::full({B, N}, -1, opts.dtype(torch::kInt32));
    auto owner = torch::full({B, M}, -1, opts.dtype(torch::kInt32));
    
    // Scratch for bids: [B, M] of packed uint64
    auto bids_packed = torch::zeros({B, M}, opts.dtype(torch::kInt64)); // 64-bit zeroed
    
    // Unassigned count (host pinned or managed?)
    // Simplest: Tensor on device, copy to host.
    auto unassigned_count = torch::full({1}, B * N, opts.dtype(torch::kInt32));
    
    int iter = 0;
    while (iter < max_iter) {
        
        // 1. Bid Phase
        int block_size = 256;
        int grid_bid = B * N; // One block per agent
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(C.scalar_type(), "auction_bid", ([&] {
            auction_bid_kernel_block<scalar_t, 4><<<grid_bid, block_size>>>(
                C.data_ptr<scalar_t>(),
                prices.data_ptr<scalar_t>(),
                assignment.data_ptr<int>(),
                (unsigned long long*)bids_packed.data_ptr<int64_t>(),
                B, N, M, epsilon
            );
        }));
        
        // 2. Assign Phase
        int grid_assign = (B * M + block_size - 1) / block_size;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(C.scalar_type(), "auction_assign", ([&] {
            auction_assign_kernel<scalar_t><<<grid_assign, block_size>>>(
                prices.data_ptr<scalar_t>(),
                assignment.data_ptr<int>(),
                owner.data_ptr<int>(),
                (unsigned long long*)bids_packed.data_ptr<int64_t>(),
                unassigned_count.data_ptr<int>(),
                B, N, M
            );
        }));
        
        // Check convergence
        iter++;
        // expensive scalar copy?
        // if (unassigned_count.item<int>() == 0) break; 
        // We typically run fixed iter or check periodically.
        // For EXACT, we must loop until 0.
        // optimization: check every K iters.
        if (iter % 10 == 0) {
            if (unassigned_count.item<int>() == 0) break;
        }
    }
    
    // Return Assignment Indices (B, N)
    // Convert to assignment matrix P? User usually expects P [B, N, N].
    // But other backends return P.
    // We should convert assignment indices to P logic in Python wrapper.
    // Here we return indices.
    
    return {assignment, prices};
}
