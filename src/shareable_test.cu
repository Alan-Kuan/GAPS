#include <unistd.h>

#include <exception>
#include <iostream>

#include <cuda.h>

#include "allocator/shareable.hpp"
#include "error.hpp"

using namespace std;

__global__ void vecAdd(int* c, int* a, int* b) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__global__ void vecMul(int* c, int* a, int* b) {
    c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
}

void sender(void);
void receiver(bool do_sum);

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << "1|2|3" << endl;
        return 1;
    }

    switch (argv[1][0]) {
    case '1':
        sender();
        return 0;
    case '2':
        receiver(true);
        return 0;
    case '3':
        receiver(false);
        return 0;
    }

    return 0;
}

void sender(void) {
    cout << "Run as a sender" << endl;
    try {
        if (cuInit(0) != CUDA_SUCCESS) throwError();

        ShareableAllocator alloc("shareable_test", 64 * 2 * sizeof(int));

        int N;
        cout << "Number of receiver = ";
        cin >> N;
        alloc.shareHandle(N);
        
        int arr[64];

        int* a = (int*) alloc.malloc(64 * sizeof(int));
        cout << "offset = " << (a - (int*) alloc.pool_base) << endl;
        for (int i = 0; i < 64; i++) {
            arr[i] = i;
            cout << arr[i] << ' ';
        }
        cout << endl;
        cudaMemcpy(a, arr, 64 * sizeof(int), cudaMemcpyHostToDevice);

        int* b = (int*) alloc.malloc(64 * sizeof(int));
        cout << "offset = " << (b - (int*) alloc.pool_base) << endl;
        for (int i = 0; i < 64; i++) {
            arr[i] = 5;
            cout << arr[i] << ' ';
        }
        cout << endl;
        cudaMemcpy(b, arr, 64 * sizeof(int), cudaMemcpyHostToDevice);

        getchar();  // remove '\n'
        getchar();  // wait

        alloc.free(a);
        alloc.free(b);
    } catch (runtime_error& err) {
        cerr << "[sender] " << err.what() << endl;
    }
}

void receiver(bool do_sum) {
    if (do_sum)
        cout << "Run as a receiver (add a and b)" << endl;
    else
        cout << "Run as a receiver (muliply a and b)" << endl;
    try {
        if (cuInit(0) != CUDA_SUCCESS) throwError();

        ShareableAllocator alloc("shareable_test");

        alloc.recvHandle();
        cout << "Handle received" << endl;

        size_t offset;
        cout << "a's offset = ";
        cin >> offset;
        // NOTE: `pool_base` is protected in alloc. To run this code, we need to
        //       temporaily change it to "public" and then compile
        int* a = (int*) alloc.pool_base + offset;

        cout << "b's offset = ";
        cin >> offset;
        int* b = (int*) alloc.pool_base + offset;

        int* c;
        cudaMalloc(&c, sizeof(int) * 64);
        int arr[64];

        if (do_sum) {
            vecAdd<<<1, 64>>>(c, a, b);
            cudaMemcpy(arr, c, sizeof(int) * 64, cudaMemcpyDeviceToHost);
        } else {
            vecMul<<<1, 64>>>(c, a, b);
            cudaMemcpy(arr, c, sizeof(int) * 64, cudaMemcpyDeviceToHost);
        }

        for (int i = 0; i < 64; i++) cout << arr[i] << ' ';
        cout << endl;

        cudaFree(c);
    } catch (runtime_error& err) {
        cerr << "[receiver] " << err.what() << endl;
    }
}
