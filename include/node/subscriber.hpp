#ifndef SUBSCRIBER_HPP
#define SUBSCRIBER_HPP

#include <functional>

#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "node/node.hpp"

class Subscriber : public Node {
public:
    typedef std::function<void(void*)> MessageHandler;

    Subscriber() = delete;
    Subscriber(const char* topic_name, const char* conf_path, const Allocator::Domain& domain);
    ~Subscriber();

    void sub(MessageHandler handler);

private:
    zenoh::Session z_session;
    zenoh::Subscriber z_subscriber;
    int domain_id;
};

#endif  // SUBSCRIBER_HPP