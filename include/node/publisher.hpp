#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <cstddef>

#include <iceoryx_posh/popo/publisher.hpp>

#include "node/node.hpp"

class Publisher : public Node {
public:
    Publisher() = delete;
    Publisher(const char* topic_name, size_t pool_size);

    void put(void* payload, size_t size);

protected:
    iox::popo::Publisher<size_t> iox_publisher;
};

#endif  // PUBLISHER_HPP