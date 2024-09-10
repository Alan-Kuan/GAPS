#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <functional>

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>

namespace nb = nanobind;

typedef nb::ndarray<nb::pytorch, nb::device::cuda> DeviceTensor;
#endif
#include <zenoh.hxx>

#include "node/node.hpp"

class Subscriber : public Node {
public:
#ifdef BUILD_PYSHOZ
    typedef std::function<void(const DeviceTensor&)> MessageHandler;
#else
    typedef std::function<void(void*, size_t)> MessageHandler;
#endif

    Subscriber() = delete;
    Subscriber(const char* topic_name, const char* llocator, size_t pool_size);
    ~Subscriber();

    void sub(MessageHandler handler);

protected:
    zenoh::Session z_session;
    zenoh::Subscriber z_subscriber;
};

#endif  // SUBSCRIBER_HPP