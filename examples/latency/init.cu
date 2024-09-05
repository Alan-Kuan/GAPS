#include "init.hpp"

__global__ void __init_data(int* arr, int tag) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    arr[tid] = tag;
}

void init_data(int* arr, size_t count, int tag) {
    dim3 grid_dim = 1;
    dim3 block_dim = count;
    if (count > 1024) {
        grid_dim = (count - 1) / 1024 + 1;
        block_dim = 1024;
    }
    __init_data<<<grid_dim, block_dim>>>(arr, tag);
}