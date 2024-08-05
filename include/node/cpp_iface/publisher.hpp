#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <cstddef>

#include "metadata.hpp"
#include "node/publisher.hpp"

class Publisher : public __Publisher {
public:
    Publisher() = delete;
    Publisher(const char* topic_name, const char* llocator,
              const Domain& domain, size_t pool_size);

    void put(void* payload, size_t size);
};

#endif  // PUBLISHER_HPP