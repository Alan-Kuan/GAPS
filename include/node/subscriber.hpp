#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <functional>

#include <zenoh.hxx>

#include "node/node.hpp"
#include "metadata.hpp"

class Subscriber : public Node {
public:
    typedef std::function<void(void*)> MessageHandler;

    Subscriber() = delete;
    Subscriber(const char* topic_name, const char* conf_path, const Domain& domain, size_t pool_size);
    ~Subscriber();

    void sub(MessageHandler handler);

private:
    zenoh::Session z_session;
    zenoh::Subscriber z_subscriber;
};

#endif  // SUBSCRIBER_HPP