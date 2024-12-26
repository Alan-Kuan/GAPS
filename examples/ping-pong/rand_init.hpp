#ifndef INIT_HPP
#define INIT_HPP

#include <curand_kernel.h>

void initRandStates(curandState* states, size_t count, unsigned long seed);
void fillRandVals(curandState* states, int* arr, size_t count);

#endif  // INIT_HPP