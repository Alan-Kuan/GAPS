#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <functional>

#include "metadata.hpp"
#include "node/subscriber.hpp"

class Subscriber : public __Subscriber {
public:
    typedef std::function<void(void*, size_t)> MessageHandler;

    Subscriber() = delete;
    Subscriber(const char* topic_name, const char* llocator,
               const Domain& domain, size_t pool_size);

    void sub(MessageHandler handler);
};

#endif  // SUBSCRIBER_HPP