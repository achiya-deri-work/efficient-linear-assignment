#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include "common.cuh"

// ----------------------------------------------------------------------
// Log-Stabilized Sinkhorn Persistent Kernel
// ----------------------------------------------------------------------

template <typename scalar_t>
__global__ void sinkhorn_persistent_kernel(
    const scalar_t* __restrict__ C,
    const scalar_t* __restrict__ log_mu,
    const scalar_t* __restrict__ log_nu,
    scalar_t* __restrict__ f,
    scalar_t* __restrict__ g,
    scalar_t* __restrict__ workspace, // [B, GridSize, Dim] for reductions? OR atomic?
                                      // Actually, for persistent kernel, we can use global mem 
                                      // with barrier.
    int B, int N, int M,
    float epsilon,
    int max_iter,
    GlobalBarrier barrier_state
) {
    // Shared Memory for Block Reduction
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float block_max;
    // Removed unused block_sum_exp
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int grid_dim = gridDim.x;
    
    // Initialize barrier (only leader)
    if (tid == 0 && bid == 0) {
        // Assume initialized from host
    }
    
    GlobalBarrier barrier = barrier_state;
    
    for (int iter = 0; iter < max_iter; iter++) {

        // -----------------------------------------------------------
        // ROW UPDATE: f = log(mu) - LSE( (-C + g) / eps )
        // -----------------------------------------------------------
        
        int total_rows = B * N;
        for (int idx = bid; idx < total_rows; idx += grid_dim) {
            int b = idx / N;
            int r = idx % N;
            
            float m_i = -INFINITY;
            float l_i = 0.0f;
            
            // Pointers for this batch
            const scalar_t* C_batch = C + b * N * M;
            scalar_t* g_batch = g + b * M;
            const scalar_t* log_mu_batch = log_mu + b * N;
            
            // Loop over M (Cols of this row)
            for (int m_idx = tid; m_idx < M; m_idx += blockDim.x) {
                float c_val = static_cast<float>(C_batch[r * M + m_idx]);
                float g_val = static_cast<float>(g_batch[m_idx]);
                
                float term = (-c_val / epsilon) + g_val;
                
                float old_m = m_i;
                float new_m = fmaxf(old_m, term);
                
                l_i = l_i * expf(old_m - new_m) + expf(term - new_m);
                m_i = new_m;
            }
            
            // Reduce
            float row_max = BlockReduce(temp_storage).Reduce(m_i, cub::Max());
            __syncthreads();
            if (tid == 0) block_max = row_max;
            __syncthreads();
            row_max = block_max;
            
            float term_exp = l_i * expf(m_i - row_max);
            float row_sum = BlockReduce(temp_storage).Sum(term_exp);
            
            if (tid == 0) {
                float lse = row_max + logf(row_sum);
                // Write global f
                f[idx] = static_cast<scalar_t>(static_cast<float>(log_mu_batch[r]) - lse);
            }
            __syncthreads();
        }
        
        barrier.sync();
        
        // -----------------------------------------------------------
        // COL UPDATE: g = log(nu) - LSE( (-C + f) / eps )
        // -----------------------------------------------------------
        
        int total_cols = B * M;
        for (int idx = bid; idx < total_cols; idx += grid_dim) {
            int b = idx / M;
            int c = idx % M;
            
            float m_i = -INFINITY;
            float l_i = 0.0f;
            
            const scalar_t* C_batch = C + b * N * M;
            scalar_t* f_batch = f + b * N;
            const scalar_t* log_nu_batch = log_nu + b * M;
            
            for (int n_idx = tid; n_idx < N; n_idx += blockDim.x) {
                float c_val = static_cast<float>(C_batch[n_idx * M + c]); // C[r, c]
                float f_val = static_cast<float>(f_batch[n_idx]);
                
                float term = (-c_val / epsilon) + f_val;
                
                float old_m = m_i;
                float new_m = fmaxf(old_m, term);
                l_i = l_i * expf(old_m - new_m) + expf(term - new_m);
                m_i = new_m;
            }
            
            float col_max = BlockReduce(temp_storage).Reduce(m_i, cub::Max());
            __syncthreads();
            if (tid == 0) block_max = col_max;
            __syncthreads();
            col_max = block_max;
            
            float term_exp = l_i * expf(m_i - col_max);
            float col_sum = BlockReduce(temp_storage).Sum(term_exp);
            
            if (tid == 0) {
                float lse = col_max + logf(col_sum);
                g[idx] = static_cast<scalar_t>(static_cast<float>(log_nu_batch[c]) - lse);
            }
            __syncthreads();
        }
        
        barrier.sync();
    }
}

// Host Wrapper
std::vector<torch::Tensor> sinkhorn_cuda_forward(
    torch::Tensor C,
    torch::Tensor log_mu,
    torch::Tensor log_nu,
    float epsilon,
    int max_iter
) {
    auto B = C.size(0);
    auto N = C.size(1);
    auto M = C.size(2);
    
    auto f = torch::zeros({B, N}, C.options());
    auto g = torch::zeros({B, M}, C.options());
    
    // Persistent Grid Logic
    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    
    // We want enough blocks to hide latency but allow sync.
    // Usually 1-4 blocks per SM.
    int blocks_per_sm = 2; // Tuneable
    int grid_size = num_sms * blocks_per_sm;
    
    // Limit grid size if N/M are small?
    // If N=128, M=128. 216 blocks is overkill.
    // 1 Block per Row is ideal if N < GridSize.
    // If N is small, Grid = N.
    long long needed = (long long)B * std::max(N, M);
    if (grid_size > needed) grid_size = (int)needed;
    if (grid_size < 1) grid_size = 1;
    
    // Barrier setup
    auto options = C.options().dtype(torch::kInt32);
    auto count = torch::zeros({1}, options);
    auto sense = torch::zeros({1}, options);
    
    GlobalBarrier barrier_host;
    barrier_host.init((unsigned int*)count.data_ptr(), (unsigned int*)sense.data_ptr(), grid_size);
    // Note: We need to pass pointers to device memory. count/sense are tensors on device.
    
    // Launch
    int block_size = 256;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(C.scalar_type(), "sinkhorn_persistent", ([&] {
        sinkhorn_persistent_kernel<scalar_t><<<grid_size, block_size>>>(
            C.data_ptr<scalar_t>(),
            log_mu.data_ptr<scalar_t>(),
            log_nu.data_ptr<scalar_t>(),
            f.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(),
            nullptr, // workspace
            B, N, M,
            epsilon,
            max_iter,
            barrier_host
        );
    }));
    
    return {f, g};
}
