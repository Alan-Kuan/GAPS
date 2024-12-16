#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <functional>
#include <string>

#include <zenoh.hxx>

#include "node/node.hpp"

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>

#include "zenoh_wrapper.hpp"

namespace nb = nanobind;

typedef nb::ndarray<nb::pytorch, nb::device::cuda> DeviceTensor;
typedef ZenohSession session_t;
#else
typedef zenoh::Session session_t;
#endif

class Subscriber : public Node {
public:
#ifdef BUILD_PYSHOZ
    typedef std::function<void(const DeviceTensor&)> MessageHandler;
#else
    typedef std::function<void(void*, size_t)> MessageHandler;
#endif

    Subscriber() = delete;
    Subscriber(const session_t& session, std::string&& topic_name,
               size_t pool_size, MessageHandler handler);
    ~Subscriber();

protected:
    std::function<void(const zenoh::Sample&)> makeCallback(
        MessageHandler handler);

    zenoh::Subscriber<void> z_subscriber;
};

#endif  // SUBSCRIBER_HPP