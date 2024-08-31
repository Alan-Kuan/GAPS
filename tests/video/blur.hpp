#ifndef BLUR_HPP
#define BLUR_HPP

#include <cstddef>

void blur(unsigned char* frame_in, unsigned char* frame_out, size_t frame_width,
          size_t frame_height, double* kernel, size_t kernel_width);

#endif  // BLUR_HPP