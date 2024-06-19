#include "error.hpp"

#include <errno.h>

#include <cstring>
#include <source_location>
#include <sstream>
#include <stdexcept>

[[noreturn]] void throwError(const char* msg, std::source_location loc) {
    std::stringstream ss;

    ss << "file: " << loc.file_name()
       << '(' << loc.line() << ") "
       << "in `" << loc.function_name() << '`';

    if (msg) ss << ": " << msg;

    throw std::runtime_error(ss.str());
}

int throwOnError(int ret, std::source_location loc) {
    if (ret < 0) throwError(strerror(errno), loc);
    return ret;
}