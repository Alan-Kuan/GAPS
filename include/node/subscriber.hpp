#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <cstddef>
#include <functional>

#include <iceoryx_posh/popo/listener.hpp>

#include "node/node.hpp"

#ifdef BUILD_PYGAPS
#include <iceoryx_posh/popo/untyped_subscriber.hpp>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

typedef nb::ndarray<nb::pytorch, nb::device::cuda> DeviceTensor;
typedef iox::popo::UntypedSubscriber iox_subscriber_t;
#else
#include <iceoryx_posh/popo/subscriber.hpp>

typedef iox::popo::Subscriber<size_t> iox_subscriber_t;
#endif

class Subscriber : public Node {
public:
#ifdef BUILD_PYGAPS
    typedef std::function<void(const DeviceTensor&)> MessageHandler;
#else
    typedef std::function<void(void*, size_t)> MessageHandler;
#endif

    Subscriber() = delete;
    Subscriber(const char* topic_name, size_t pool_size, int msg_queue_cap_exp,
               MessageHandler handler);
    ~Subscriber();

protected:
    iox_subscriber_t iox_subscriber;
    iox::popo::Listener iox_listener;
    MessageHandler handler;
    MessageQueueHeader* mq_header;

    static void onSampleReceived(iox_subscriber_t* iox_subscriber,
                                 Subscriber* self);
};

#endif  // SUBSCRIBER_HPP