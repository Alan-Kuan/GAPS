#include "node/publisher.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <zenoh-pico/config.h>
#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "allocator/shareable.hpp"
#include "error.hpp"
#include "metadata.hpp"

Publisher::Publisher(const char* topic_name, const char* llocator,
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

void Publisher::put(void* payload, size_t size) {
    if (!payload) throwError("Payload was not provided");
    if (size == 0) return;

    size_t offset = this->allocator->malloc(size);
    if (offset == -1) throwError("No free space in the pool");
    void* addr = (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);
    this->allocator->copyTo(addr, payload, size);

    // get next available index as message id
    MessageQueueHeader* mq_header = getMessageQueueHeader(
        getDomainMap(getTlsfHeader(getTopicHeader(this->shm_base))));
    size_t msg_id =
        std::atomic_ref<size_t>(mq_header->next).fetch_add(1) % kMaxMessageNum;
    MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header, msg_id);

    // free the payload's space if it hasn't been freed
    if (mq_entry->avail) {
        this->allocator->free(mq_entry->offset);
    }

    // NOTE: though `taken_num` and `avail` should be atomic referenced, it's
    // okay because
    //       no other process will access these variables at this moment
    mq_entry->taken_num = 0;
    mq_entry->offset = offset;
    mq_entry->size = size;
    mq_entry->avail = 1 << this->domain_idx;

    // notify subscribers with the message ID
    zenoh::BytesView msg((void*) &msg_id, sizeof(size_t));
    if (!this->z_publisher.put(msg)) {
        throwError("Zenoh failed to send a message");
    }
}