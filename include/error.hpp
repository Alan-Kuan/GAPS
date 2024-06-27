#ifndef ERROR_HPP
#define ERROR_HPP

#include <source_location>

#include <cuda.h>

[[noreturn]] void throwError(const char* msg = nullptr, std::source_location loc = std::source_location::current());

int throwOnError(int ret, std::source_location loc = std::source_location::current());
void throwOnErrorCuda(CUresult res, std::source_location loc = std::source_location::current());

#endif  // ERROR_HPP