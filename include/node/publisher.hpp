#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <cstddef>
#include <string>

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>

namespace nb = nanobind;

typedef nb::ndarray<nb::pytorch, nb::device::cuda> DeviceTensor;
#endif
#include <zenoh.hxx>

#include "node/node.hpp"

class Publisher : public Node {
public:
    Publisher() = delete;
    Publisher(const zenoh::Session& z_session, std::string&& topic_name,
              size_t pool_size);

#ifdef BUILD_PYSHOZ
    void copyTensor(DeviceTensor& dst, const nb::ndarray<nb::pytorch>& src);
    DeviceTensor malloc(size_t ndim, nb::tuple shape, nb::tuple dtype_tup,
                        bool clean);
    void put(const DeviceTensor& tensor);
#else
    void* malloc(size_t size);
    void put(void* payload, size_t size);
#endif

protected:
    zenoh::Publisher z_publisher;
};

#endif  // PUBLISHER_HPP