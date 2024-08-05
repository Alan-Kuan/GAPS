#include "node/subscriber.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <zenoh-pico/config.h>
#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "allocator/shareable.hpp"
#include "metadata.hpp"

__Subscriber::__Subscriber(const char* topic_name, const char* llocator,
                           const Domain& domain, size_t pool_size)
        : Node(topic_name, pool_size, domain.getId()),
          z_session(nullptr),
          z_subscriber(nullptr) {
    zenoh::Config config;
    config.insert(Z_CONFIG_MODE_KEY, Z_CONFIG_MODE_PEER);
    config.insert(Z_CONFIG_LISTEN_KEY, llocator);
    this->z_session =
        zenoh::expect<zenoh::Session>(zenoh::open(std::move(config)));

    switch (domain.dev_type) {
    case DeviceType::kGPU:
        this->allocator = (Allocator*) new ShareableAllocator(
            (TopicHeader*) this->shm_base, true);
        break;
    }
}

__Subscriber::~__Subscriber() {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    std::atomic_ref<uint32_t>(topic_header->sub_count)--;
}