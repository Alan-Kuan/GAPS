#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <cstddef>
#include <functional>

#include <iceoryx_posh/popo/listener.hpp>
#include <iceoryx_posh/popo/subscriber.hpp>

#include "node/node.hpp"

class Subscriber : public Node {
public:
    typedef std::function<void(void*, size_t)> MessageHandler;

    Subscriber() = delete;
    Subscriber(const char* topic_name, size_t pool_size,
               MessageHandler handler);
    ~Subscriber();

protected:
    iox::popo::Subscriber<size_t> iox_subscriber;
    iox::popo::Listener iox_listener;
    MessageHandler handler;
    MessageQueueHeader* mq_header;

    static void onSampleReceived(iox::popo::Subscriber<size_t>* iox_subscriber,
                                 Subscriber* self);
};

#endif  // SUBSCRIBER_HPP