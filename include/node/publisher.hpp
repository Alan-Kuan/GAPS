#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <cstddef>

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>
namespace nb = nanobind;
#endif
#include <zenoh.hxx>

#include "metadata.hpp"
#include "node/node.hpp"

class Publisher : public Node {
public:
    Publisher() = delete;
    Publisher(const char* topic_name, const char* llocator,
              const Domain& domain, size_t pool_size);

#ifdef BUILD_PYSHOZ
    void put(const nb::ndarray<>& tensor);
#else
    void put(void* payload, size_t size);
#endif

protected:
    zenoh::Session z_session;
    zenoh::Publisher z_publisher;
};

#endif  // PUBLISHER_HPP