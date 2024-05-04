#include <sys/wait.h>
#include <unistd.h>

#include <exception>
#include <iostream>

#include <cuda.h>

#include "allocator/shareable.hpp"
#include "error.hpp"

using namespace std;

__global__ void vecMul(int* arr, int mul) {
    arr[threadIdx.x] *= mul;
}

void sender(void);
void receiver(void);

int main(void) {
    switch (fork()) {
    case -1:
        cerr << "fork failed" << endl;
        return 1;
    case 0:
        receiver();
        return 0;
    default:
        sender();
    }
    wait(nullptr);

    return 0;
}

void sender(void) {
    try {
        if (cuInit(0) != CUDA_SUCCESS) throwError();

        ShareableAllocator alloc("shareable_test", 64);
        
        int arr[64];
        int* d_arr = (int*) alloc.malloc(64);

        for (int i = 0; i < 64; i++) arr[i] = i;
        cudaMemcpy(d_arr, arr, sizeof(int) * 64, cudaMemcpyHostToDevice);

        alloc.shareHandle(1);
    } catch (runtime_error& err) {
        cerr << "[sender] " << err.what() << endl;
    }
}

void receiver(void) {
    try {
        if (cuInit(0) != CUDA_SUCCESS) throwError();

        ShareableAllocator alloc("shareable_test");

        alloc.recvHandle();

        int arr[64];
        int* d_arr = (int*) alloc.malloc(64);

        vecMul<<<1, 64>>>(d_arr, 5);
        cudaMemcpy(arr, d_arr, sizeof(int) * 64, cudaMemcpyDeviceToHost);

        for (int i = 0; i < 64; i++) cout << arr[i] << ' ';
        cout << endl;
    } catch (runtime_error& err) {
        cerr << "[receiver] " << err.what() << endl;
    }
}