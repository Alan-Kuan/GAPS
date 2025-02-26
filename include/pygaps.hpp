#ifndef PYGAPS_HPP
#define PYGAPS_HPP

#include <cstdint>

enum class Dtype : uint8_t {
    int8,
    int16,
    int32,
    int64,
    uint8,
    float16,
    float32,
};

#endif  // PYGAPS_HPP