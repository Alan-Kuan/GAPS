#include "blur.hpp"

#include <cstddef>

__global__ void conv(unsigned char* frame_in, unsigned char* frame_out,
                     size_t frame_width, size_t frame_height, double* kernel,
                     size_t kernel_width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= frame_height || col >= frame_width) return;

    size_t half_kernel_width = kernel_width / 2;

    for (int c = 0; c < 3; c++) {
        double sum = 0;
        for (size_t i = 0; i < kernel_width; i++) {
            for (size_t j = 0; j < kernel_width; j++) {
                int row_target = row + i - half_kernel_width;
                int col_target = col + j - half_kernel_width;
                if (row_target < 0 || row_target >= frame_height ||
                    col_target < 0 || col_target >= frame_width)
                    continue;
                sum +=
                    kernel[i * kernel_width + j] *
                    frame_in[(row_target * frame_width + col_target) * 3 + c];
            }
        }
        frame_out[(row * frame_width + col) * 3 + c] = (unsigned char) sum;
    }
}

void blur(unsigned char* frame_in, unsigned char* frame_out, size_t frame_width,
          size_t frame_height, double* kernel, size_t kernel_width) {
    dim3 grid_dim{316 / 32 + 1, 600 / 32 + 1};
    dim3 block_dim{32, 32};

    conv<<<grid_dim, block_dim>>>(frame_in, frame_out, frame_width,
                                  frame_height, kernel, kernel_width);
}