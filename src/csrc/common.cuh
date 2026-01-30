#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ----------------------------------------------------------------------
// Global Barrier for Persistent Kernels
// ----------------------------------------------------------------------
// Allows synchronizing across all blocks in the grid.
// Requires that the grid size does not exceed the number of resident blocks
// the GPU can support (or use cooperative launch).

struct GlobalBarrier {
    unsigned int* count;
    unsigned int* sense;
    int expected_blocks;

    __host__ __device__ void init(unsigned int* count_in, unsigned int* sense_in, int size) {
        count = count_in;
        sense = sense_in;
        expected_blocks = size;
    }

    __device__ void sync() {
        __threadfence();
        
        // Elect a leader for the block (tid 0)
        int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        
        if (tid == 0) {
            unsigned int my_sense = atomicLoad(sense);
            unsigned int arrive = atomicAdd(count, 1);
            
            if (arrive == expected_blocks - 1) {
                // Last block to arrive
                atomicStore(count, 0);
                atomicStore(sense, !my_sense);
            } else {
                // Wait for sense to flip
                while (atomicLoad(sense) == my_sense) {
                    __threadfence_block(); // Spin
                    // PTX 'nanosleep' could be used here for efficiency
                    #if __CUDA_ARCH__ >= 700
                    __nanosleep(100); 
                    #endif
                }
            }
        }
        __syncthreads();
    }
    
    // Helpers for atomics (relaxed consistency is usually fine for barrier flags, 
    // but we need release/acquire for visibility)
    // Here we use default Sequential Consistency for safety.
    __device__ unsigned int atomicLoad(unsigned int* addr) {
        return atomicAdd(addr, 0); // Hacky volatile load or proper atomic
        // Ideally: return volatile *addr; but caching issues.
        // atomicAdd(0) forces global memory return.
    }
    
    __device__ void atomicStore(unsigned int* addr, unsigned int val) {
        atomicExch(addr, val);
    }
};
