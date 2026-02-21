#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;

// ----------------------------------------------------------------------
// Unified Flash Dual Ascent Kernel
// Supports:
// - InputT: float, half_t, bfloat16_t
// - MathT:  tfloat32_t (TF32), half_t (FP16), bfloat16_t (BF16)
// ----------------------------------------------------------------------

// Helper to deduce MMA Op and Value Type
template <typename MathT> struct FlashSpecs;

// TF32 Math (Default for Float)
template <> struct FlashSpecs<cutlass::tfloat32_t> { 
    using Op = SM80_16x8x8_F32TF32TF32F32_TN; 
    using ValT = cutlass::tfloat32_t;
    using BlockK = Int<32>; 
};

// FP16 Math
template <> struct FlashSpecs<cutlass::half_t> {
    using Op = SM80_16x8x16_F32F16F16F32_TN;
    using ValT = cutlass::half_t;
    using BlockK = Int<64>; 
};

// BF16 Math
template <> struct FlashSpecs<cutlass::bfloat16_t> {
    using Op = SM80_16x8x16_F32BF16BF16F32_TN;
    using ValT = cutlass::bfloat16_t;
    using BlockK = Int<64>; 
};

template <typename InputT, typename MathT> 
__global__ void flash_dual_ascent_kernel(
    const InputT* __restrict__ Q_ptr,      // [B, N, D]
    const InputT* __restrict__ K_ptr,      // [B, M, D]
    const InputT* __restrict__ mu_ptr,     // [B, N]
    const InputT* __restrict__ nu_ptr,     // [B, M]
    InputT* __restrict__ alpha_ptr,        // [B, N] (Read-Modify-Write)
    const InputT* __restrict__ beta_ptr,   // [B, M]
    int B, int N, int M, int D,
    float epsilon,
    float* __restrict__ grad_P_ptr         // [B, N] Accumulator (Always Float)
) {
    using Specs = FlashSpecs<MathT>;
    using MMA_Op = typename Specs::Op;
    using MMA_ValT = typename Specs::ValT;
    using BlockK = typename Specs::BlockK;
    
    using BlockM = Int<128>;
    using BlockN = Int<128>;
    
    // Tiled MMA
    using TiledMma = TiledMMA<
        MMA_Atom<MMA_Op>,
        Layout<Shape<_2, _2, _1>>,  // 4 Warps
        Tile<BlockM, BlockN, BlockK> 
    >;

    int batch_idx = blockIdx.y;

    // Global Tensors
    Tensor Q = make_tensor(make_gmem_ptr(Q_ptr + batch_idx * N * D), make_shape(N, D), make_stride(D, _1{}));
    Tensor K = make_tensor(make_gmem_ptr(K_ptr + batch_idx * M * D), make_shape(M, D), make_stride(D, _1{}));
    
    // Shared Memory
    using SmemLayoutQ = Layout<Shape<BlockM, BlockK>, Stride<BlockK, _1>>;
    using SmemLayoutK = Layout<Shape<BlockN, BlockK>, Stride<BlockK, _1>>;
    
    extern __shared__ char smem_buf[];
    Tensor sQ = make_tensor(make_smem_ptr(smem_buf), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(smem_buf + size(sQ)*sizeof(InputT)), SmemLayoutK{}); 

    // Copy Atom (AutoVectorizingCopy for safety with generic inputs)
    using CopyAtom = Copy_Atom<AutoVectorizingCopy, InputT>;
    auto tiled_copy = make_tiled_copy(
        CopyAtom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<1>{}))
    );
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
    
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tC = make_tensor<float>(make_shape(BlockM{}, BlockN{})); // Accum Always Float
    auto tCrC = thr_mma.partition_fragment_C(tC); 
    
    int M_tiles = (M + BlockN{} - 1) / BlockN{};
    int D_tiles = (D + BlockK{} - 1) / BlockK{}; 

    // Loop M
    for (int m_tile = 0; m_tile < M_tiles; ++m_tile) {
        clear(tCrC);

        // Loop D
        for (int d_tile = 0; d_tile < D_tiles; ++d_tile) {
             // 1. Load Q sub-tile
             Tensor gQ_tile = local_tile(Q, make_tile(BlockM{}, BlockK{}), make_coord(blockIdx.x, d_tile));
             Tensor tQgQ = thr_copy.partition_S(gQ_tile);
             Tensor tQsQ = thr_copy.partition_D(sQ);
             copy(tiled_copy, tQgQ, tQsQ);
             
             // 2. Load K sub-tile
             Tensor gK_tile = local_tile(K, make_tile(BlockN{}, BlockK{}), make_coord(m_tile, d_tile));
             Tensor tKgK = thr_copy.partition_S(gK_tile);
             Tensor tKsK = thr_copy.partition_D(sK);
             copy(tiled_copy, tKgK, tKsK);
             
             __syncthreads();
             
             // 3. Gemm
             auto tCrA = thr_mma.partition_fragment_A(sQ); 
             auto tCrB = thr_mma.partition_fragment_B(sK); 
             
             // Cast to MathT in Registers
             auto tOrA = make_fragment_like<MMA_ValT>(tCrA);
             auto tOrB = make_fragment_like<MMA_ValT>(tCrB);
             
             copy(tCrA, tOrA); 
             copy(tCrB, tOrB);
             
             gemm(tiled_mma, tCrC, tOrA, tOrB, tCrC);
             
             __syncthreads(); 
        }
        
        // Epilogue
        Tensor cIdentity = make_identity_tensor(Shape<BlockM, BlockN>{});
        auto tCid = thr_mma.partition_C(cIdentity);
        
        for (int i = 0; i < size(tCrC); ++i) {
             float c_val = tCrC(i); 
             auto coord = tCid(i);
             int m_in_tile = get<0>(coord); 
             int n_in_tile = get<1>(coord);
             
             int global_row = blockIdx.x * BlockM{} + m_in_tile;
             int global_col = m_tile * BlockN{} + n_in_tile;
             
             if (global_row < N && global_col < M) {
                  float my_alpha = (float)alpha_ptr[batch_idx * N + global_row];
                  float my_beta = (float)beta_ptr[batch_idx * M + global_col];
                  
                  // Score = alpha + beta - (Q.K^T). 
                  // If Q.K^T is positive (Implicit), we use + c_val.
                  float T_val = my_alpha + my_beta + c_val;
                  if (T_val > 0.0f) {
                     float p_val = T_val / epsilon; 
                     atomicAdd(&grad_P_ptr[batch_idx * N + global_row], p_val);
                  }
             }
        }
    }
}

template <typename T>
__global__ void flash_update_alpha_kernel(
    T* __restrict__ alpha_ptr,        
    const float* __restrict__ grad_P_ptr, 
    const T* __restrict__ mu_ptr,     
    int N, 
    int step_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float p_sum = (float)grad_P_ptr[idx];
        float mu_val = (float)mu_ptr[idx];
        float lr = 1.0f / (step_idx + 1);
        float grad = mu_val - p_sum;
        alpha_ptr[idx] = (T)((float)alpha_ptr[idx] + lr * grad);
    }
}

// ----------------------------------------------------------------------
// Dispatcher
// ----------------------------------------------------------------------

// Helper Mappings
// InputT: float -> float, half->cutlass::half_t, bfloat16->bfloat16_t
// We use generic dispatch from AT_DISPATCH. scalar_t maps to InputT.

// MathT mapping:
// string "tf32" -> cutlass::tfloat32_t
// string "fp16" -> cutlass::half_t
// string "bf16" -> cutlass::bfloat16_t
// string "fp32" -> float

template <typename InputT>
void launch_kernel(
    std::string math_mode,
    torch::Tensor Q, torch::Tensor K, torch::Tensor mu, torch::Tensor nu,
    torch::Tensor alpha, torch::Tensor beta,
    int B, int N, int M, int D,
    float epsilon, int max_iter,
    float* grad_P_ptr
) {
    if (math_mode == "tf32") {
        if constexpr (std::is_same_v<InputT, float>) { // TF32 valid for float input mostly, but we can cast
            // Ensure compatibility. 
            using MathT = cutlass::tfloat32_t;
            using Traits = MMA_Traits<MathT>; // Check if exists
            size_t smem_size = (128*32 + 128*32) * sizeof(InputT);
            
            dim3 block(128);            
            int GridX = (N + 128 - 1) / 128;
            dim3 grid(GridX, B);

            if (smem_size > 48 * 1024) cudaFuncSetAttribute(flash_dual_ascent_kernel<InputT, MathT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            for(int i=0; i<max_iter; ++i) {
                cudaMemset(grad_P_ptr, 0, B*N*sizeof(float));
                flash_dual_ascent_kernel<InputT, MathT><<<grid, block, smem_size>>>(
                    (const InputT*)Q.data_ptr(), (const InputT*)K.data_ptr(),
                    (const InputT*)mu.data_ptr(), (const InputT*)nu.data_ptr(),
                    (InputT*)alpha.data_ptr(), (const InputT*)beta.data_ptr(),
                    B, N, M, D, epsilon, grad_P_ptr
                );
                
                int threads = 256;
                flash_update_alpha_kernel<InputT><<<(N+255)/256, 256>>>(
                     (InputT*)alpha.data_ptr(), grad_P_ptr, (const InputT*)mu.data_ptr(), N, i
                );
            }
        } else {
             // TF32 usually means Float input logic. If Input is Half, maybe not TF32?
             // Actually you can load Half -> Convert TF32 -> MMA. Valid.
             // Allow it.
            using MathT = cutlass::tfloat32_t;
            size_t smem_size = (128*32 + 128*32) * sizeof(InputT); 
            dim3 block(128);            
            int GridX = (N + 128 - 1) / 128;
            dim3 grid(GridX, B);
            
            if (smem_size > 48 * 1024) cudaFuncSetAttribute(flash_dual_ascent_kernel<InputT, MathT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            for(int i=0; i<max_iter; ++i) {
                cudaMemset(grad_P_ptr, 0, B*N*sizeof(float));
                flash_dual_ascent_kernel<InputT, MathT><<<grid, block, smem_size>>>(
                    (const InputT*)Q.data_ptr(), (const InputT*)K.data_ptr(),
                    (const InputT*)mu.data_ptr(), (const InputT*)nu.data_ptr(),
                    (InputT*)alpha.data_ptr(), (const InputT*)beta.data_ptr(),
                    B, N, M, D, epsilon, grad_P_ptr
                );
                flash_update_alpha_kernel<InputT><<<(N+255)/256, 256>>>(
                     (InputT*)alpha.data_ptr(), grad_P_ptr, (const InputT*)mu.data_ptr(), N, i
                );
            }
        }
    } else if (math_mode == "fp16") {
        using MathT = cutlass::half_t;
        // BlockK is 64 for FP16
        size_t smem_size = (128*64 + 128*64) * sizeof(InputT);
        dim3 block(128);            
        int GridX = (N + 128 - 1) / 128;
        dim3 grid(GridX, B);
        if (smem_size > 48 * 1024) cudaFuncSetAttribute(flash_dual_ascent_kernel<InputT, MathT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        for(int i=0; i<max_iter; ++i) {
                cudaMemset(grad_P_ptr, 0, B*N*sizeof(float));
                flash_dual_ascent_kernel<InputT, MathT><<<grid, block, smem_size>>>(
                    (const InputT*)Q.data_ptr(), (const InputT*)K.data_ptr(),
                    (const InputT*)mu.data_ptr(), (const InputT*)nu.data_ptr(),
                    (InputT*)alpha.data_ptr(), (const InputT*)beta.data_ptr(),
                    B, N, M, D, epsilon, grad_P_ptr
                );
                flash_update_alpha_kernel<InputT><<<(N+255)/256, 256>>>(
                     (InputT*)alpha.data_ptr(), grad_P_ptr, (const InputT*)mu.data_ptr(), N, i
                );
        }
    } else if (math_mode == "bf16") {
        // Only if input is bf16? Or can we mix? 
        // Mix is fine.
        using MathT = cutlass::bfloat16_t;
        size_t smem_size = (128*64 + 128*64) * sizeof(InputT);
        dim3 block(128);            
        int GridX = (N + 128 - 1) / 128;
        dim3 grid(GridX, B);
        if (smem_size > 48 * 1024) cudaFuncSetAttribute(flash_dual_ascent_kernel<InputT, MathT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        for(int i=0; i<max_iter; ++i) {
                cudaMemset(grad_P_ptr, 0, B*N*sizeof(float));
                flash_dual_ascent_kernel<InputT, MathT><<<grid, block, smem_size>>>(
                    (const InputT*)Q.data_ptr(), (const InputT*)K.data_ptr(),
                    (const InputT*)mu.data_ptr(), (const InputT*)nu.data_ptr(),
                    (InputT*)alpha.data_ptr(), (const InputT*)beta.data_ptr(),
                    B, N, M, D, epsilon, grad_P_ptr
                );
                flash_update_alpha_kernel<InputT><<<(N+255)/256, 256>>>(
                     (InputT*)alpha.data_ptr(), grad_P_ptr, (const InputT*)mu.data_ptr(), N, i
                );
        }
    }
}


std::vector<torch::Tensor> flash_dual_ascent_dispatch(
    torch::Tensor Q, torch::Tensor K, torch::Tensor mu, torch::Tensor nu,
    float epsilon, int max_iter, 
    std::string math_mode
) {
    auto B = Q.size(0);
    auto N = Q.size(1);
    auto M = K.size(1);
    auto D = Q.size(2);
    auto options = Q.options();
    auto alpha = torch::zeros({B, N}, options);
    auto beta = torch::zeros({B, M}, options);
    auto grad_P = torch::zeros({B, N}, options.dtype(torch::kFloat32)); 
    float* grad_P_ptr = grad_P.data_ptr<float>();

    // Dispatch InputT
    if (Q.scalar_type() == torch::kFloat32) {
        launch_kernel<float>(math_mode, Q, K, mu, nu, alpha, beta, B, N, M, D, epsilon, max_iter, grad_P_ptr);
    } else if (Q.scalar_type() == torch::kHalf) {
        launch_kernel<cutlass::half_t>(math_mode, Q, K, mu, nu, alpha, beta, B, N, M, D, epsilon, max_iter, grad_P_ptr);
    } else if (Q.scalar_type() == torch::kBFloat16) {
        launch_kernel<cutlass::bfloat16_t>(math_mode, Q, K, mu, nu, alpha, beta, B, N, M, D, epsilon, max_iter, grad_P_ptr);
    } else {
        TORCH_CHECK(false, "Unsupported Input Type");
    }

    return {alpha, beta};
}

// Bindings aliases using the dispatcher
std::vector<torch::Tensor> flash_dual_ascent_v3_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor mu, torch::Tensor nu,
    float epsilon, int max_iter
) {
    return flash_dual_ascent_dispatch(Q, K, mu, nu, epsilon, max_iter, "fp16"); // V3 default
}

std::vector<torch::Tensor> flash_dual_ascent_v4_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor mu, torch::Tensor nu,
    float epsilon, int max_iter
) {
    return flash_dual_ascent_dispatch(Q, K, mu, nu, epsilon, max_iter, "tf32"); // V4 default
}
