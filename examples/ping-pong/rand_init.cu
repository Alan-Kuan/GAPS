#include "rand_init.hpp"

#include <curand_kernel.h>

__global__ void __initRandStates(curandState* states, unsigned long seed) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, states + tid);
}

__global__ void __fillRandVals(curandState* states, int* arr) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    arr[tid] = curand_uniform(states + tid) * 10;
}

void initRandStates(curandState* states, size_t count, unsigned long seed) {
    dim3 grid_dim = 1;
    dim3 block_dim = count;
    if (count > 1024) {
        grid_dim = (count - 1) / 1024 + 1;
        block_dim = 1024;
    }
    __initRandStates<<<grid_dim, block_dim>>>(states, seed);
    cudaDeviceSynchronize();
}

void fillRandVals(curandState* states, int* arr, size_t count) {
    dim3 grid_dim = 1;
    dim3 block_dim = count;
    if (count > 1024) {
        grid_dim = (count - 1) / 1024 + 1;
        block_dim = 1024;
    }
    __fillRandVals<<<grid_dim, block_dim>>>(states, arr);
    cudaDeviceSynchronize();
}