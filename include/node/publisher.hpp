#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <cstddef>

#include <zenoh.hxx>

#include "metadata.hpp"
#include "node/node.hpp"

class Publisher : public Node {
public:
    Publisher() = delete;
    Publisher(const char* topic_name, const char* conf_path,
              const Domain& domain, size_t pool_size);

    void put(void* payload, size_t size);

private:
    zenoh::Session z_session;
    zenoh::Publisher z_publisher;
};

#endif  // PUBLISHER_HPP