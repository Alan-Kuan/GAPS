#include "error.hpp"

#include <errno.h>

#include <cstring>
#include <source_location>
#include <sstream>
#include <stdexcept>

#include <cuda.h>

[[noreturn]] void throwError(const char* msg, std::source_location loc) {
    std::stringstream ss;

    if (msg) ss << msg;
    ss << std::endl;
    ss << " - File: " << loc.file_name() << std::endl;
    ss << " - Line: " << loc.line() << std::endl;
    ss << " - Function: " << loc.function_name() << std::endl;

    throw std::runtime_error(ss.str());
}

int throwOnError(int ret, std::source_location loc) {
    if (ret < 0) throwError(strerror(errno), loc);
    return ret;
}

void throwOnErrorCuda(CUresult res, std::source_location loc) {
    if (res == CUDA_SUCCESS) return;
    const char* msg;
    cuGetErrorString(res, &msg);
    throwError(msg, loc);
}