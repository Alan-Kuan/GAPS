#include "init.hpp"

#include <curand.h>
#include <curand_kernel.h>

__global__ void __setup_rand_states(curandState* states, unsigned long seed) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, states + tid);
}

__global__ void __init_data(curandState* states, int* arr) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    arr[tid] = curand_uniform(states + tid) * 10;
}

void setup_rand_states(curandState* states, size_t count, unsigned long seed) {
    dim3 grid_dim = 1;
    dim3 block_dim = count;
    if (count > 1024) {
        grid_dim = (count - 1) / 1024 + 1;
        block_dim = 1024;
    }
    __setup_rand_states<<<grid_dim, block_dim>>>(states, seed);
    cudaDeviceSynchronize();
}

void init_data(curandState* states, int* arr, size_t count) {
    dim3 grid_dim = 1;
    dim3 block_dim = count;
    if (count > 1024) {
        grid_dim = (count - 1) / 1024 + 1;
        block_dim = 1024;
    }
    __init_data<<<grid_dim, block_dim>>>(states, arr);
    cudaDeviceSynchronize();
}