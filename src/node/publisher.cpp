#include "node/publisher.hpp"

#include <cstddef>
#include <cstring>

#include <zenoh-pico/config.h>
#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "allocator/shareable.hpp"
#include "metadata.hpp"

__Publisher::__Publisher(const char* topic_name, const char* llocator,
                         const Domain& domain, size_t pool_size)
        : Node(topic_name, pool_size, domain.getId()),
          z_session(nullptr),
          z_publisher(nullptr) {
    zenoh::Config config;
    config.insert(Z_CONFIG_MODE_KEY, Z_CONFIG_MODE_PEER);
    config.insert(Z_CONFIG_LISTEN_KEY, llocator);
    this->z_session =
        zenoh::expect<zenoh::Session>(zenoh::open(std::move(config)));

    char z_topic_name[kMaxTopicNameLen + 6];
    sprintf(z_topic_name, "shoz/%s", topic_name);
    this->z_publisher = zenoh::expect<zenoh::Publisher>(
        z_session.declare_publisher(z_topic_name));

    switch (domain.dev_type) {
    case DeviceType::kGPU:
        this->allocator =
            (Allocator*) new ShareableAllocator((TopicHeader*) this->shm_base);
        break;
    }
}