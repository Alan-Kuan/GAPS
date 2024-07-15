#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <functional>

#include <zenoh.hxx>

#include "metadata.hpp"
#include "node/node.hpp"

class Subscriber : public Node {
public:
    typedef std::function<void(void*, size_t)> MessageHandler;

    Subscriber() = delete;
    Subscriber(const char* topic_name, const char* llocator,
               const Domain& domain, size_t pool_size);
    ~Subscriber();

    void sub(MessageHandler handler);

private:
    zenoh::Session z_session;
    zenoh::Subscriber z_subscriber;
};

#endif  // SUBSCRIBER_HPP