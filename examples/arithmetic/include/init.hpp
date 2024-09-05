#ifndef INIT_HPP
#define INIT_HPP

#include <curand_kernel.h>

void setup_rand_states(curandState* states, size_t count, unsigned long seed);
void init_data(curandState* states, int* arr, size_t count);

#endif  // INIT_HPP