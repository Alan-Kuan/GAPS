#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <cstddef>

#include <nanobind/ndarray.h>

#include "metadata.hpp"
#include "node/publisher.hpp"

namespace nb = nanobind;

class Publisher : public __Publisher {
public:
    Publisher() = delete;
    Publisher(const char* topic_name, const char* llocator,
              const Domain& domain, size_t pool_size);

    void put(const nb::ndarray<>& tensor);
};

#endif  // PUBLISHER_HPP