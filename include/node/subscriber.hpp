#ifndef __SUBSCRIBER_HPP
#define __SUBSCRIBER_HPP

#include <zenoh.hxx>

#include "metadata.hpp"
#include "node/node.hpp"

class __Subscriber : public Node {
public:
    __Subscriber() = delete;
    __Subscriber(const char* topic_name, const char* llocator,
                 const Domain& domain, size_t pool_size);
    ~__Subscriber();

protected:
    zenoh::Session z_session;
    zenoh::Subscriber z_subscriber;
};

#endif  // __SUBSCRIBER_HPP