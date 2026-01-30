#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/fast_math.h>

#include "common.cuh"

// ----------------------------------------------------------------------
// CUTLASS Sinkhorn Kernel
// ----------------------------------------------------------------------
// Uses cutlass::Array for vectorization and cutlass::fast_exp.
// Maintains the Persistent Kernel architecture.

template <typename T, int VecSize>
__global__ void sinkhorn_cutlass_kernel(
    const T* __restrict__ C_ptr,
    const T* __restrict__ log_mu_ptr,
    const T* __restrict__ log_nu_ptr,
    T* __restrict__ f_ptr,
    T* __restrict__ g_ptr,
    int B, int N, int M,
    float epsilon,
    int max_iter,
    GlobalBarrier barrier_state
) {
    // Shared Memory for block reduction
    // We only need simple reductions for row/col.
    // For large N, we do multiple passes or block-stride loops.
    // Here we stick to a simple strategy: Each thread processes elements,
    // then block reduce.
    
    // Note: To use cutlass::Array, we need aligned pointers.
    // If not aligned, we fall back to scalar? Pytorch tensors are usually aligned.
    
    using VecT = cutlass::Array<T, VecSize>;
    using AccessT = cutlass::AlignedArray<T, VecSize>; // Assumes alignment!
    
    // int tid = threadIdx.x; // Unused in this kernel logic now (we use gid)
    // int bid = blockIdx.x;
    int grid_size = gridDim.x;
    
    // Total number of rows/cols in the Batch
    int total_rows = B * N;
    int total_cols = B * M;
    
    // Persistent Loop
    // We iterate 'iter' from 0 to max_iter.
    // Inside, we loop over all rows/cols assigned to this block.
    
    // Precompute epsilon inv (float for math)
    float eps_inv_f = 1.0f / epsilon;
    
    // Correct Grid-Stride Loop using GID (Global Thread ID)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int iter = 0; iter < max_iter; ++iter) {
    
        // -----------------------------------------------------------
        // ROW UPDATE
        // -----------------------------------------------------------
        for (int row_idx = gid; row_idx < total_rows; row_idx += stride) {
            int b = row_idx / N;
            int i = row_idx % N;
            
            const T* row_C = C_ptr + (b * N * M) + (i * M);
            const T* vec_g = g_ptr + (b * M);
            
            // 1. Find Max
            float max_val = -1e30f;
            
            // Vectorized Loop
            int m_vec_limit = (M / VecSize) * VecSize;
            
            for (int j = 0; j < m_vec_limit; j += VecSize) {
                VecT c_val = *reinterpret_cast<const VecT*>(row_C + j);
                VecT g_val = *reinterpret_cast<const VecT*>(vec_g + j);
                
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < VecSize; ++k) {
                    float val = static_cast<float>(g_val[k]) - (static_cast<float>(c_val[k]) * eps_inv_f);
                    max_val = (val > max_val) ? val : max_val;
                }
            }
            for (int j = m_vec_limit; j < M; ++j) {
                 float val = static_cast<float>(vec_g[j]) - (static_cast<float>(row_C[j]) * eps_inv_f);
                 max_val = (val > max_val) ? val : max_val;
            }
            
            // 2. Sum Exp
            float sum_exp = 0.0f;
            for (int j = 0; j < m_vec_limit; j += VecSize) {
                VecT c_val = *reinterpret_cast<const VecT*>(row_C + j);
                VecT g_val = *reinterpret_cast<const VecT*>(vec_g + j);
                
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < VecSize; ++k) {
                    float val = static_cast<float>(g_val[k]) - (static_cast<float>(c_val[k]) * eps_inv_f);
                    sum_exp += cutlass::fast_exp(val - max_val);
                }
            }
            for (int j = m_vec_limit; j < M; ++j) {
                float val = static_cast<float>(vec_g[j]) - (static_cast<float>(row_C[j]) * eps_inv_f);
                sum_exp += cutlass::fast_exp(val - max_val);
            }
            
            float lse = max_val + logf(sum_exp);
            f_ptr[row_idx] = static_cast<T>(static_cast<float>(log_mu_ptr[row_idx]) - lse);
        }
        
        // Synchronize all threads in grid
        if (grid_size > 1) { // grid_size is gridDim.x
           barrier_state.sync(); 
        }
        
        // -----------------------------------------------------------
        // COL UPDATE
        // -----------------------------------------------------------
        // -----------------------------------------------------------
        // COL UPDATE
        // -----------------------------------------------------------
        // Vectorized: Each thread handles 'VecSize' columns.
        int vec_cols = total_cols / VecSize; // Assumes M multiple of VecSize
        
        for (int vec_idx = gid; vec_idx < vec_cols; vec_idx += stride) {
            int b = vec_idx / (M / VecSize);
            int vec_j = vec_idx % (M / VecSize);
            
            const T* mat_C_base = C_ptr + (b * N * M);
            const T* vec_f = f_ptr + (b * N);
            
            T* vec_g_ptr_loc = g_ptr + (b * M);
            VecT* vec_g_ptr_vec = reinterpret_cast<VecT*>(vec_g_ptr_loc);
            
            const T* log_nu_vec_loc = log_nu_ptr + (b * M);
            const VecT* log_nu_vec_ptr = reinterpret_cast<const VecT*>(log_nu_vec_loc);
            
            // Per-element accumulators
            float max_val[VecSize];
            float sum_exp[VecSize];
            
            #pragma unroll
            for (int k = 0; k < VecSize; ++k) max_val[k] = -1e30f;
            
            // 1. Find Max (Vectorized over Rows)
            for (int i = 0; i < N; ++i) {
                // Load Row i, Col chunk vec_j
                const VecT* row_C_vec_ptr = reinterpret_cast<const VecT*>(mat_C_base + i * M);
                VecT c_vec = row_C_vec_ptr[vec_j];
                
                float f_val = static_cast<float>(vec_f[i]);
                
                #pragma unroll
                for (int k = 0; k < VecSize; ++k) {
                    float val = f_val - (static_cast<float>(c_vec[k]) * eps_inv_f);
                    max_val[k] = (val > max_val[k]) ? val : max_val[k];
                }
            }
            
            #pragma unroll
            for (int k = 0; k < VecSize; ++k) sum_exp[k] = 0.0f;
            
            // 2. Sum Exp
            for (int i = 0; i < N; ++i) {
                const VecT* row_C_vec_ptr = reinterpret_cast<const VecT*>(mat_C_base + i * M);
                VecT c_vec = row_C_vec_ptr[vec_j];
                float f_val = static_cast<float>(vec_f[i]);
                
                #pragma unroll
                for (int k = 0; k < VecSize; ++k) {
                    float val = f_val - (static_cast<float>(c_vec[k]) * eps_inv_f);
                    sum_exp[k] += cutlass::fast_exp(val - max_val[k]);
                }
            }
            
            // Store
            VecT final_g;
            VecT l_nu = log_nu_vec_ptr[vec_j];
            
            #pragma unroll
            for (int k = 0; k < VecSize; ++k) {
                float lse = max_val[k] + logf(sum_exp[k]);
                final_g[k] = static_cast<T>(static_cast<float>(l_nu[k]) - lse);
            }
            
            vec_g_ptr_vec[vec_j] = final_g;
        }
        
        if (grid_size > 1) {
            barrier_state.sync();
        }
    }
}

// Wrapper to launch
std::vector<torch::Tensor> sinkhorn_cutlass_forward(
    torch::Tensor C,
    torch::Tensor log_mu,
    torch::Tensor log_nu,
    float epsilon,
    int max_iter
) {
    auto B = C.size(0);
    auto N = C.size(1);
    auto M = C.size(2);
    
    auto opts = C.options();
    auto f = torch::zeros({B, N}, opts);
    auto g = torch::zeros({B, M}, opts);
    
    // Global Barrier
    int grid_size = 128; // Can we optimize this?
    if (B * N < grid_size) grid_size = B * N;
    if (grid_size < 1) grid_size = 1;
    
    // Allocate barrier
    auto barrier_opts = torch::TensorOptions().dtype(torch::kInt32).device(C.device());
    auto count = torch::zeros({1}, barrier_opts);
    auto sense = torch::zeros({1}, barrier_opts);
    
    GlobalBarrier barrier_host;
    barrier_host.init((unsigned int*)count.data_ptr<int>(), (unsigned int*)sense.data_ptr<int>(), grid_size);
    
    int block_size = 256; // One warp per row? No, here one thread per row. 
                          // Wait, the above kernel is ST (Single Thread) per row/col?
                          // "Stripe over all rows". Loop: 'for row_idx = bid'.
                          // This means BLOCKs stripe. THREADS? 
                          // Ah, the kernel above uses 'tid' NOWHERE in the logic for distributing work!
                          // It assumes 'bid' distributes work, but threads in block?
                          // The code above: "for (int row_idx = bid; ...)"
                          // This means the WHOLE BLOCK does row_idx? No, 'row_idx = bid' implies Block ID.
                          // But threads?
                          // The 'sinkhorn_cutlass_kernel' above is written as if it's Single-Thread-Per-Task.
                          // That is very inefficient for GPUs.
                          
    // CORRECTION: We should use Grid-Stride Loop with Thread ID.
    // row_idx = bid * block_size + tid.
    
    // I need to rewrite the loop logic in the kernel slightly before saving.
    // Let's fix it in the string content below.
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(C.scalar_type(), "sinkhorn_cutlass", ([&] {
        sinkhorn_cutlass_kernel<scalar_t, 4><<<grid_size, block_size>>>(
            C.data_ptr<scalar_t>(),
            log_mu.data_ptr<scalar_t>(),
            log_nu.data_ptr<scalar_t>(),
            f.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(),
            B, N, M,
            epsilon,
            max_iter,
            barrier_host
        );
    }));
    
    return {f, g};
}
