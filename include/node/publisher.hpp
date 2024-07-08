#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <cstddef>

#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "node/node.hpp"

class Publisher : public Node {
public:
    Publisher() = delete;
    Publisher(const char* topic_name, const char* conf_path,
              const Allocator::Domain& domain, size_t pool_size);
    ~Publisher();

    void put(void* payload, size_t size);

private:
    zenoh::Session z_session;
    zenoh::Publisher z_publisher;
    int domain_id;
};

#endif  // PUBLISHER_HPP