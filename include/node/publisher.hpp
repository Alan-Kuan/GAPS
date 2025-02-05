#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <cstddef>
#include <string>

#include <zenoh.hxx>

#include "node/node.hpp"

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>

#include "pyshoz.hpp"
#include "zenoh_wrapper.hpp"

namespace nb = nanobind;

typedef nb::ndarray<nb::pytorch, nb::device::cuda> DeviceTensor;
typedef ZenohSession session_t;
#else
typedef zenoh::Session session_t;
#endif

class Publisher : public Node {
public:
    Publisher() = delete;
    Publisher(const session_t& session, std::string&& topic_name,
              size_t pool_size, int msg_queue_cap_exp);

#ifdef BUILD_PYSHOZ
    DeviceTensor malloc(nb::tuple shape, Dtype dtype);
    void put(const DeviceTensor& tensor);
#else
    void* malloc(size_t size);
    void put(void* payload, size_t size);
#endif

protected:
    zenoh::Publisher z_publisher;
};

#endif  // PUBLISHER_HPP