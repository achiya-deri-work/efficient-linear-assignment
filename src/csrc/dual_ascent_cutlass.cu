#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>

#include "common.cuh"

// ----------------------------------------------------------------------
// CuTe Dual Ascent Persistent Kernel
// ----------------------------------------------------------------------

namespace dual_ascent_cute {

using namespace cute;

// Persistent Kernel using CuTe for Layouts and Tiling
template <typename T>
__global__ void dual_ascent_persistent_kernel_cute(
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
    // 1. Problem Shape & Identity
    // We treat the problem as Batch of (N, M). 
    // Grid: We map blocks to (Rows / BlockM, B).
    // Or simpler: Flatten B*N rows.
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    // We need a persistent loop over the Grid if grid_size < B*N.
    // However, for CuTe tiling, it's easier to map 1 CTA -> 1 Tile of Rows.
    // Let's assume Grid covers the problem or we loop stride.
    
    // Config: Block size 256.
    // Each thread processes vector elements?
    // Let's use CuTe Tensors.
    
    // Global Memory Tensors
    // C: (M, N, B) (Row-Major in phys is (M, N) if stride is (1, M)?)
    // Input C is torch (B, N, M). Stride (N*M, M, 1).
    // Let's view as (M, N, B).
    // Stride: (1, M, N*M).
    auto tensor_C = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N, B), make_stride(_1{}, M, N*M));
    auto tensor_alpha = make_tensor(make_gmem_ptr(alpha_ptr), make_shape(N, B), make_stride(_1{}, N));
    // Mu is same as Alpha layout
    auto tensor_mu = make_tensor(make_gmem_ptr(mu_ptr), make_shape(N, B), make_stride(_1{}, N));
    
    // Beta: (M, B). Stride (1, M)? No, Beta is (B, M). Stride (M, 1).
    // Beta (B, M). Phys: B major? No usually RowMajor (B, M) means B is outer.
    // tensor[b, m].
    // Wait, let's stick to simple flattening. 
    // Total Rows = B*N.
    
    float eps_inv = 1.0f / epsilon;
    float step = epsilon * 0.5f;

    // --------------------------------------------------------
    // Persistent Loop
    // --------------------------------------------------------
    for (int iter = 0; iter < max_iter; ++iter) {
        
        // --- ROW UPDATE (Partitioned by Grid) ---
        // Each thread block takes a chunk of Rows.
        // We use typical CUDA grid stride for robustness.
        
        int total_rows = B * N;
        for (int row_idx = bid * blockDim.x + tid; row_idx < total_rows; row_idx += gridDim.x * blockDim.x) {
            int b = row_idx / N;
            int i = row_idx % N;
            
            // Slice Row i of Batch b from C => C( :, i, b ) -> vector of size M
            auto row_C = tensor_C(_ , i, b);  // (M)
            auto val_beta = make_tensor(make_gmem_ptr(beta_ptr + b * M), make_shape(M), make_stride(_1{}));
            
            float my_alpha = static_cast<float>(alpha_ptr[row_idx]);
            float my_mu = static_cast<float>(mu_ptr[row_idx]);
            
            float sum_P = 0.0f;
            
            // Vectorized loop over M using CuTe?
            // "Copy" with TiledCopy/Gmem Tiled Copy?
            // Or just logical iteration.
            // With M dynamic, we loop.
            
            // Manual loop for now is safest with dynamic M, but we can verify alignment.
            // If we use `coalesce` we rely on CTA-collective. But this loop is per-thread (Scalar).
            // Optimization: Each Thread handles ONE ROW?
            // If N is large (4096), M is large (4096).
            // One thread looping 4096 is OK (4k cycles).
            
            // Can we Vectorize?
            // row_C is contiguous (Stride 1). val_beta is contiguous.
            // Reinterpret as float4/half2?
            // CuTe helps with `recast`.
            
            // Vectorized Loading with CuTe
            // Recast Tensors to vector types to force 128-bit loads
            // Assuming T=float, we use float4. If half, we use 8 elements?
            // Let's stick to 128-bit alignment assumption.
            
            using VecType = uint4; // 128-bit load container
            // Reinterpret the tensor view.
            auto row_C_vec = recast<VecType>(row_C);    // (M/4)
            auto beta_vec  = recast<VecType>(val_beta); // (M/4)
            
            int M_vec = size(row_C_vec);

            for (int j=0; j<M_vec; ++j) {
                // Load 128-bits
                VecType c_v = row_C_vec(j);
                VecType b_v = beta_vec(j);
                
                // Explode back to T
                T* c_ptr = reinterpret_cast<T*>(&c_v);
                T* b_ptr = reinterpret_cast<T*>(&b_v);
                
                // Process 4/8 elements
                // Assuming sizeof(VecType) / sizeof(T) elements
                int k_limit = sizeof(VecType) / sizeof(T);
                
                for (int k=0; k<k_limit; ++k) {
                    float val = (my_alpha + static_cast<float>(b_ptr[k]) - static_cast<float>(c_ptr[k])) * eps_inv;
                     if (val > 0.0f) sum_P += val;
                }
            }
            // Handle remainder? CuTe Recast usually asserts divisibility.
            // For now assuming M % 4 == 0 (Checked in utils).
            
            // Update Alpha
            float grad = my_mu - sum_P;
            alpha_ptr[row_idx] = static_cast<T>(my_alpha + step * grad);
        }
        
        // Sync Blocks
        if (gridDim.x > 1) barrier_state.sync();
        
        // --- COL UPDATE ---
        // Parallelize over Cols (B*M).
        int total_cols = B * M;
        for (int col_idx = bid * blockDim.x + tid; col_idx < total_cols; col_idx += gridDim.x * blockDim.x) {
            int b = col_idx / M;
            int j = col_idx % M;
            
            // C(:, :, b). Column j -> C(j, :, b) -> Stride is M! (Non-contiguous)
            // Input C is (B, N, M).
            // C[b, i, j].
            // Fix loop: We want Sum_i P_ij.
            // Iterate i. C access is C[b, i, j]. C stride is M.
            // Non-contiguous memory access!
            // This is the bottleneck.
            
            float my_beta = static_cast<float>(beta_ptr[col_idx]);
            float my_nu = static_cast<float>(nu_ptr[col_idx]);
            
            // We need Alpha vector (B, N) -> alpha[b, :] (Contiguous).
            const T* alpha_vec = alpha_ptr + b * N;
            // C column: C[b, 0, j], C[b, 1, j]... Stride M.
            const T* c_col_ptr = C_ptr + (b * N * M) + j;
            
            float sum_P = 0.0f;
            
            for (int i=0; i<N; ++i) {
                float a_val = static_cast<float>(alpha_vec[i]);
                float c_val = static_cast<float>(c_col_ptr[i * M]);
                
                float val = (a_val + my_beta - c_val) * eps_inv;
                if (val > 0.0f) sum_P += val;
            }
            
            float grad = my_nu - sum_P;
            beta_ptr[col_idx] = static_cast<T>(my_beta + step * grad);
        }
        
        if (gridDim.x > 1) barrier_state.sync();
    }
}

// Host Launcher
std::vector<torch::Tensor> forward(
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
    
    // Persistent Grid
    // Total Work ~ max(B*N, B*M).
    // We want enough blocks to occupy GPU but allow sync.
    // 108 SMs * 4 blocks = 432 blocks.
    int grid_size = 320; 
    int block_size = 256;
    
    auto barrier_opts = torch::TensorOptions().dtype(torch::kInt32).device(C.device());
    auto count = torch::zeros({1}, barrier_opts);
    auto sense = torch::zeros({1}, barrier_opts);
    
    GlobalBarrier barrier_host;
    barrier_host.init((unsigned int*)count.data_ptr<int>(), (unsigned int*)sense.data_ptr<int>(), grid_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BFLOAT16(C.scalar_type(), "dual_ascent_cute", ([&] {
        dual_ascent_persistent_kernel_cute<scalar_t><<<grid_size, block_size>>>(
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

} // namespace

// Legacy Wrapper for binding
std::vector<torch::Tensor> dual_ascent_cutlass_forward(
    torch::Tensor C,
    torch::Tensor mu,
    torch::Tensor nu,
    float epsilon,
    int max_iter
) {
    return dual_ascent_cute::forward(C, mu, nu, epsilon, max_iter);
}
