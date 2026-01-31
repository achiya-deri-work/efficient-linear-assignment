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
    
    // Half2 Helpers
    bool is_half = std::is_same<T, cutlass::half_t>::value;
    
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
            
            float sum_P = 0.0f;
            
            if (is_half && VecSize % 2 == 0) {
                 // PACKED HALF2 PATH
                 int m_half2_limit = M / 2;
                 const __half2* row_C_h2 = reinterpret_cast<const __half2*>(row_C);
                 const __half2* vec_beta_h2 = reinterpret_cast<const __half2*>(vec_beta);
                 __half2 alpha_h2 = __float2half2_rn(my_alpha);
                 __half2 eps_inv_h2 = __float2half2_rn(eps_inv_f);
                 __half2 zero_h2 = __float2half2_rn(0.0f);
                 
                 for (int j = 0; j < m_half2_limit; ++j) {
                     __half2 c_val = row_C_h2[j];
                     __half2 b_val = vec_beta_h2[j];
                     
                     // P = ReLU((alpha + beta - C)/eps)
                     __half2 diff = __hadd2(alpha_h2, __hsub2(b_val, c_val));
                     __half2 val = __hmul2(diff, eps_inv_h2);
                     
                     // ReLU
                     // There is no __hmax2 in standard CUDA < 11? Or use fallback.
                     // __hmax2 is available on Ampere+ or via polyfill. 
                     // Safe approach: convert back to float or check support.
                     // Assuming recent CUDA (User has 12.8):
                     #if __CUDA_ARCH__ >= 530
                        val = __hmax2(val, zero_h2);
                        // Sum accumulation needs float
                        float2 f2 = __half22float2(val);
                        sum_P += f2.x + f2.y;
                     #else
                        float2 f2 = __half22float2(val);
                        sum_P += fmaxf(f2.x, 0.0f) + fmaxf(f2.y, 0.0f);
                     #endif
                 }
                 // Handle remainder if M is odd? (Likely M is multiple of 8)
            } else {
                // FLOAT PATH
                int m_vec_limit = (M / VecSize) * VecSize;
                
                for (int j = 0; j < m_vec_limit; j += VecSize) {
                    VecT c_val = *reinterpret_cast<const VecT*>(row_C + j);
                    VecT b_val = *reinterpret_cast<const VecT*>(vec_beta + j);
                    
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < VecSize; ++k) {
                        float val = (my_alpha + static_cast<float>(b_val[k]) - static_cast<float>(c_val[k])) * eps_inv_f;
                        if (val > 0.0f) sum_P += val;
                    }
                }
                for (int j = m_vec_limit; j < M; ++j) {
                    float val = (my_alpha + static_cast<float>(vec_beta[j]) - static_cast<float>(row_C[j])) * eps_inv_f;
                    if (val > 0.0f) sum_P += val;
                }
            }
            
            // Gradient Ascent
            float grad = my_mu - sum_P;
            alpha_ptr[row_idx] = static_cast<T>(my_alpha + step_size * grad);
        }
        
        // Sync
        if (gridDim.x > 1) barrier_state.sync();
        
        // -----------------------------------------------------------
        // COL UPDATE: beta
        // -----------------------------------------------------------
        int vec_cols = total_cols / VecSize; 
        
        for (int vec_idx = gid; vec_idx < vec_cols; vec_idx += stride) {
            int b = vec_idx / (M / VecSize);
            int vec_j = vec_idx % (M / VecSize);
            int j_start = vec_j * VecSize;
            
            const T* mat_C_base = C_ptr + (b * N * M);
            const T* vec_alpha = alpha_ptr + (b * N);
            
            VecT* vec_beta_ptr = reinterpret_cast<VecT*>(beta_ptr + (b * M));
            const VecT* vec_nu_ptr = reinterpret_cast<const VecT*>(nu_ptr + (b * M));
            
            VecT my_beta_vec = vec_beta_ptr[vec_j];
            VecT my_nu_vec = vec_nu_ptr[vec_j];
            
            float sum_P[VecSize];
            #pragma unroll
            for (int k = 0; k < VecSize; ++k) sum_P[k] = 0.0f;
            
            // Loop Rows (i)
            // Can we pack here? It's harder because we vectorizing over COLS, but iterating ROWS.
            // C is Row-Major (b, i, j).
            // loading C[i, j_start ... j_start+V] is contiguous.
            // If T=half, VecSize=4 (typical for float), then we load 4 halfs (64 bits). 
            // We can process them as 2x __half2.
            
            if (is_half && VecSize == 4) {
                 // Optimized Half2 Loop for VecSize=4
                 __half2 eps_inv_h2 = __float2half2_rn(eps_inv_f);
                 __half2 zero_h2 = __float2half2_rn(0.0f);
                 
                 // Reinterpret local accums as float pairs? No, we sum to float.
                 // We have 4 lanes (k=0..3).
                 
                 for (int i = 0; i < N; ++i) {
                     float a_val = static_cast<float>(vec_alpha[i]);
                     __half2 alpha_h2 = __float2half2_rn(a_val); // {a, a}
                     
                     const __half2* row_C_h2 = reinterpret_cast<const __half2*>(mat_C_base + i * M + j_start);
                     __half2 c1 = row_C_h2[0]; // j, j+1
                     __half2 c2 = row_C_h2[1]; // j+2, j+3
                     
                     // Get beta as half2
                     // my_beta_vec is Array<half, 4>.
                     const __half2* beta_h2_ptr = reinterpret_cast<const __half2*>(&my_beta_vec);
                     __half2 b1 = beta_h2_ptr[0];
                     __half2 b2 = beta_h2_ptr[1];
                     
                     // Calc
                     __half2 diff1 = __hadd2(alpha_h2, __hsub2(b1, c1));
                     __half2 val1 = __hmul2(diff1, eps_inv_h2);
                     #if __CUDA_ARCH__ >= 530
                     val1 = __hmax2(val1, zero_h2);
                     #endif 
                     float2 f1 = __half22float2(val1);
                     sum_P[0] += f1.x; sum_P[1] += f1.y;

                     __half2 diff2 = __hadd2(alpha_h2, __hsub2(b2, c2));
                     __half2 val2 = __hmul2(diff2, eps_inv_h2);
                     #if __CUDA_ARCH__ >= 530
                     val2 = __hmax2(val2, zero_h2);
                     #endif
                     float2 f2 = __half22float2(val2);
                     sum_P[2] += f2.x; sum_P[3] += f2.y;
                 }
            } else {
                // Check if aligned
                for (int i = 0; i < N; ++i) {
                    float a_val = static_cast<float>(vec_alpha[i]);
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
            }
            
            // Update Beta
            VecT new_beta_vec;
            #pragma unroll
            for (int k = 0; k < VecSize; ++k) {
                float grad = static_cast<float>(my_nu_vec[k]) - sum_P[k];
                float update = static_cast<float>(my_beta_vec[k]) + step_size * grad;
                new_beta_vec[k] = static_cast<T>(update);
            }
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
