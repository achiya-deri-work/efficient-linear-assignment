#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include "common.cuh"

// ----------------------------------------------------------------------
// L2-Regularized Dual Ascent Persistent Kernel
// ----------------------------------------------------------------------

template <typename scalar_t>
__global__ void dual_ascent_persistent_kernel(
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ mu,
    const scalar_t* __restrict__ nu,
    scalar_t* __restrict__ alpha,
    scalar_t* __restrict__ beta,
    int B, int N, int M,
    float epsilon,
    int max_iter,
    GlobalBarrier barrier_state
) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float block_sum;
    __shared__ float block_count;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int grid_dim = gridDim.x;
    
    GlobalBarrier barrier = barrier_state;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // -----------------------------------------------------------
        // ROW UPDATE: alpha
        // T = alpha + beta - C
        // Active = T > 0
        // -----------------------------------------------------------
        
        int total_rows = B * N;
        for (int idx = bid; idx < total_rows; idx += grid_dim) {
            int b = idx / N;
            int r = idx % N;
            
            float alpha_val = static_cast<float>(alpha[idx]);
            float target = static_cast<float>(mu[idx]);
            
            const scalar_t* C_batch = C + b * N * M;
            scalar_t* beta_batch = beta + b * M;
            
            float sum_t = 0.0f;
            float count = 0.0f;
            
            for (int c = tid; c < M; c += blockDim.x) {
                float c_val = static_cast<float>(C_batch[r * M + c]);
                float beta_val = static_cast<float>(beta_batch[c]);
                
                float T = alpha_val + beta_val - c_val;
                if (T > 0) {
                    sum_t += T;
                    count += 1.0f;
                }
            }
            
            // Reduce
            float row_sum_t = BlockReduce(temp_storage).Sum(sum_t);
            __syncthreads();
            if (tid == 0) block_sum = row_sum_t;
            __syncthreads();
            
            float row_count = BlockReduce(temp_storage).Sum(count);
            __syncthreads();
            if (tid == 0) block_count = row_count;
            __syncthreads();
            
            if (tid == 0) {
                float current_sum = block_sum / epsilon;
                float grad = block_count / epsilon;
                grad = (grad < 1e-6f) ? 1e-6f : grad;
                
                float delta = (target - current_sum) / grad;
                alpha[idx] = static_cast<scalar_t>(alpha_val + delta);
            }
            __syncthreads();
        }
        
        barrier.sync();
        
        // -----------------------------------------------------------
        // COL UPDATE: beta
        // -----------------------------------------------------------
        
        int total_cols = B * M;
        for (int idx = bid; idx < total_cols; idx += grid_dim) {
            int b = idx / M;
            int c = idx % M;
            
            float beta_val = static_cast<float>(beta[idx]);
            float target = static_cast<float>(nu[idx]);
            
            const scalar_t* C_batch = C + b * N * M;
            scalar_t* alpha_batch = alpha + b * N;
            
            float sum_t = 0.0f;
            float count = 0.0f;
            
            for (int r = tid; r < N; r += blockDim.x) {
                float c_val = static_cast<float>(C_batch[r * M + c]); // Strided
                float alpha_val = static_cast<float>(alpha_batch[r]);
                
                float T = alpha_val + beta_val - c_val;
                if (T > 0) {
                    sum_t += T;
                    count += 1.0f;
                }
            }
            
            // Reduce
            float col_sum_t = BlockReduce(temp_storage).Sum(sum_t);
            __syncthreads();
            if (tid == 0) block_sum = col_sum_t;
            __syncthreads();
            
            float col_count = BlockReduce(temp_storage).Sum(count);
            __syncthreads();
            if (tid == 0) block_count = col_count;
            __syncthreads();
            
            if (tid == 0) {
                float current_sum = block_sum / epsilon;
                float grad = block_count / epsilon;
                grad = (grad < 1e-6f) ? 1e-6f : grad;
                
                float delta = (target - current_sum) / grad;
                beta[idx] = static_cast<scalar_t>(beta_val + delta);
            }
            __syncthreads();
        }
        
        barrier.sync();
    }
}

std::vector<torch::Tensor> dual_ascent_cuda_forward(
    torch::Tensor C,
    torch::Tensor mu,
    torch::Tensor nu,
    float epsilon,
    int max_iter
) {
    auto B = C.size(0);
    auto N = C.size(1);
    auto M = C.size(2);
    
    auto alpha = torch::zeros({B, N}, C.options());
    auto beta = torch::zeros({B, M}, C.options());
    
    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    int grid_size = num_sms * 2;
    long long needed = (long long)B * std::max(N, M);
    if (grid_size > needed) grid_size = (int)needed;
    if (grid_size < 1) grid_size = 1;
    
    auto options = C.options().dtype(torch::kInt32);
    auto count = torch::zeros({1}, options);
    auto sense = torch::zeros({1}, options);
    
    GlobalBarrier barrier_host;
    barrier_host.init((unsigned int*)count.data_ptr(), (unsigned int*)sense.data_ptr(), grid_size);
    
    int block_size = 256;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(C.scalar_type(), "dual_ascent_persistent", ([&] {
        dual_ascent_persistent_kernel<scalar_t><<<grid_size, block_size>>>(
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
