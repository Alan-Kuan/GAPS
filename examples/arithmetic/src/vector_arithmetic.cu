#include "vector_arithmetic.hpp"

#include <cstddef>

__global__ void __vecAdd(int* c, int* a, int* b) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__global__ void __vecMul(int* c, int* a, int* b) {
    c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
}

void vecAdd(int* c, int* a, int* b, size_t count) {
    __vecAdd<<<1, count>>>(c, a, b);
}

void vecMul(int* c, int* a, int* b, size_t count) {
    __vecMul<<<1, count>>>(c, a, b);
}