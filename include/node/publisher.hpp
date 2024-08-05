#ifndef __PUBLISHER_HPP
#define __PUBLISHER_HPP

#include <cstddef>

#include <zenoh.hxx>

#include "metadata.hpp"
#include "node/node.hpp"

class __Publisher : public Node {
public:
    __Publisher() = delete;
    __Publisher(const char* topic_name, const char* llocator,
                const Domain& domain, size_t pool_size);

protected:
    zenoh::Session z_session;
    zenoh::Publisher z_publisher;
};

#endif  // __PUBLISHER_HPP