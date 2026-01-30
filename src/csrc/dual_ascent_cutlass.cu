#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/fast_math.h>

#include "common.cuh"

// ----------------------------------------------------------------------
// CUTLASS Dual Ascent Kernel (L2 Regularized)
// ----------------------------------------------------------------------
// P = ReLU( (mu + nu - C) / epsilon + offset )? 
// Actually L2-OT formulation:
// Primal: min sum(C*P) + eps/2 * ||P||^2
// Dual: max sum(mu*alpha) + sum(nu*beta) - 0.5/eps * ||ReLU(alpha + beta - C)||^2
// P_optimal = 1/eps * ReLU(alpha + beta - C)
//
// Gradients:
// d_alpha_i = mu_i - sum_j P_ij
// d_beta_j  = nu_j - sum_i P_ij
//
// Update:
// alpha += step * (mu - row_sum(P))
// beta  += step * (nu - col_sum(P))
//
// We iterate this.

template <typename T, int VecSize>
__global__ void dual_ascent_cutlass_kernel(
    const T* __restrict__ C_ptr,
    const T* __restrict__ mu_ptr,
    const T* __restrict__ nu_ptr,
    T* __restrict__ alpha_ptr,
    T* __restrict__ beta_ptr,
    int B, int N, int M,
    float epsilon,
    int max_iter,
    GlobalBarrier barrier_state
) {
    using VecT = cutlass::Array<T, VecSize>;
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    int total_rows = B * N;
    int total_cols = B * M;
    
    float eps_inv_f = 1.0f / epsilon;
    float step_size = epsilon * 0.5f; 
    
    for (int iter = 0; iter < max_iter; ++iter) {
    
        // -----------------------------------------------------------
        // ROW UPDATE: alpha
        // -----------------------------------------------------------
        for (int row_idx = gid; row_idx < total_rows; row_idx += stride) {
            int b = row_idx / N;
            int i = row_idx % N;
            
            const T* row_C = C_ptr + (b * N * M) + (i * M);
            const T* vec_beta = beta_ptr + (b * M);
            float my_alpha = static_cast<float>(alpha_ptr[row_idx]);
            float my_mu = static_cast<float>(mu_ptr[row_idx]);
            
            // Compute sum(P_i)
            float sum_P = 0.0f;
            
            int m_vec_limit = (M / VecSize) * VecSize;
            
            for (int j = 0; j < m_vec_limit; j += VecSize) {
                VecT c_val = *reinterpret_cast<const VecT*>(row_C + j);
                VecT b_val = *reinterpret_cast<const VecT*>(vec_beta + j);
                
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < VecSize; ++k) {
                    // P = ReLU( (alpha + beta - C)/eps )
                    float val = (my_alpha + static_cast<float>(b_val[k]) - static_cast<float>(c_val[k])) * eps_inv_f;
                    if (val > 0.0f) sum_P += val;
                }
            }
            for (int j = m_vec_limit; j < M; ++j) {
                float val = (my_alpha + static_cast<float>(vec_beta[j]) - static_cast<float>(row_C[j])) * eps_inv_f;
                if (val > 0.0f) sum_P += val;
            }
            
            // Gradient Ascent
            // grad = mu - sum_P
            float grad = my_mu - sum_P;
            alpha_ptr[row_idx] = static_cast<T>(my_alpha + step_size * grad);
        }
        
        // Sync
        if (gridDim.x > 1) barrier_state.sync();
        
        // -----------------------------------------------------------
        // COL UPDATE: beta
        // -----------------------------------------------------------
        // -----------------------------------------------------------
        // COL UPDATE: beta
        // -----------------------------------------------------------
        // We vectorize over M. Each thread handles 'VecSize' columns.
        int vec_cols = total_cols / VecSize; // Assumes M, total_cols multiple of VecSize
        
        for (int vec_idx = gid; vec_idx < vec_cols; vec_idx += stride) {
            int b = vec_idx / (M / VecSize);
            int vec_j = vec_idx % (M / VecSize);
            int j_start = vec_j * VecSize;
            
            const T* mat_C_base = C_ptr + (b * N * M);
            const T* vec_alpha = alpha_ptr + (b * N);
            
            // Load beta, nu vectors
            VecT* vec_beta_ptr = reinterpret_cast<VecT*>(beta_ptr + (b * M));
            const VecT* vec_nu_ptr = reinterpret_cast<const VecT*>(nu_ptr + (b * M));
            
            VecT my_beta_vec = vec_beta_ptr[vec_j];
            VecT my_nu_vec = vec_nu_ptr[vec_j];
            
            // Per-element accumulators
            float sum_P[VecSize];
            #pragma unroll
            for (int k = 0; k < VecSize; ++k) sum_P[k] = 0.0f;
            
            // Loop Rows (i)
            for (int i = 0; i < N; ++i) {
                float a_val = static_cast<float>(vec_alpha[i]);
                
                // Load chunk of C: C[i, j_start ... j_start+V]
                // Pointer math: Base + i*M + j_start.
                // Cast to VecT.
                const VecT* row_C_vec_ptr = reinterpret_cast<const VecT*>(mat_C_base + i * M);
                VecT c_vec = row_C_vec_ptr[vec_j];
                
                #pragma unroll
                for (int k = 0; k < VecSize; ++k) {
                    float c_val = static_cast<float>(c_vec[k]);
                    float b_val = static_cast<float>(my_beta_vec[k]);
                    
                    float val = (a_val + b_val - c_val) * eps_inv_f;
                    if (val > 0.0f) sum_P[k] += val;
                }
            }
            
            // Update Beta
            VecT new_beta_vec;
            #pragma unroll
            for (int k = 0; k < VecSize; ++k) {
                float grad = static_cast<float>(my_nu_vec[k]) - sum_P[k];
                float update = static_cast<float>(my_beta_vec[k]) + step_size * grad;
                new_beta_vec[k] = static_cast<T>(update);
            }
            
            // Store
            vec_beta_ptr[vec_j] = new_beta_vec;
        }
        
        // Sync
        if (gridDim.x > 1) barrier_state.sync();
    }
}

std::vector<torch::Tensor> dual_ascent_cutlass_forward(
    torch::Tensor C,
    torch::Tensor mu,
    torch::Tensor nu,
    float epsilon,
    int max_iter
) {
    auto B = C.size(0);
    auto N = C.size(1);
    auto M = C.size(2);
    
    auto opts = C.options();
    auto alpha = torch::zeros({B, N}, opts);
    auto beta = torch::zeros({B, M}, opts);
    
    int grid_size = 128;
    int block_size = 256;
    
    if (B * N < grid_size) grid_size = B * N;
    if (grid_size < 1) grid_size = 1;
    
    // Allocate barrier
    auto barrier_opts = torch::TensorOptions().dtype(torch::kInt32).device(C.device());
    auto count = torch::zeros({1}, barrier_opts);
    auto sense = torch::zeros({1}, barrier_opts);
    
    GlobalBarrier barrier_host;
    barrier_host.init((unsigned int*)count.data_ptr<int>(), (unsigned int*)sense.data_ptr<int>(), grid_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(C.scalar_type(), "dual_ascent_cutlass", ([&] {
        dual_ascent_cutlass_kernel<scalar_t, 4><<<grid_size, block_size>>>(
            C.data_ptr<scalar_t>(),
            mu.data_ptr<scalar_t>(),
            nu.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            B, N, M,
            epsilon,
            max_iter,
            barrier_host
        );
    }));
    
    return {alpha, beta};
}
