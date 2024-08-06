#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <functional>

#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>

#include "metadata.hpp"
#include "node/subscriber.hpp"

namespace nb = nanobind;

class Subscriber : public __Subscriber {
public:
    typedef std::function<void(
        const nb::ndarray<nb::pytorch, nb::device::cuda>&)>
        MessageHandler;

    Subscriber() = delete;
    Subscriber(const char* topic_name, const char* llocator,
               const Domain& domain, size_t pool_size);

    void sub(MessageHandler handler);
};

#endif  // SUBSCRIBER_HPP