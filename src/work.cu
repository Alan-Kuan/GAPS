#include "work.hpp"

#include <cstddef>
#include <iostream>

__global__ void vecMul(int* arr, int mul) {
    arr[threadIdx.x] *= mul;
}

void initAndCopyDataToHost(int* buf, int count) {
    int* arr;
    int* arr_dev;
    std::size_t size = sizeof(int) * count;

    arr = new int[count];
    cudaMalloc(&arr_dev, size);

    for (int i = 0; i < count; i++) arr[i] = i;
    cudaMemcpy(arr_dev, arr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(buf, arr_dev, size, cudaMemcpyDeviceToHost);

    cudaFree(arr_dev);
    delete[] arr;
}

void copyDataToDeviceAndRun(int* buf, int count) {
    int* arr;
    int* arr_dev;
    std::size_t size = sizeof(int) * count;

    arr = new int[count];
    cudaMalloc(&arr_dev, size);

    cudaMemcpy(arr_dev, buf, size, cudaMemcpyHostToDevice);
    vecMul<<<1, count>>>(arr_dev, 2);
    cudaMemcpy(arr, arr_dev, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; i++) {
        std::cout << arr[i] << ' ';
    }
    std::cout << std::endl;

    cudaFree(arr_dev);
    delete[] arr;
}