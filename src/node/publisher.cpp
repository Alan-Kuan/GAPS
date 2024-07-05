#include "node/publisher.hpp"

#include <alloca.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "allocator/shareable.hpp"
#include "error.hpp"
#include "metadata.hpp"

Publisher::Publisher(const char* topic_name, const char* conf_path, const Domain& domain, size_t pool_size)
        : Node(topic_name, pool_size, domain.getId()),
          z_session(nullptr),
          z_publisher(nullptr) {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    std::atomic_ref<uint32_t>(topic_header->interest_count)++;

    auto config = zenoh::expect<zenoh::Config>(zenoh::config_from_file(conf_path));
    this->z_session = zenoh::expect<zenoh::Session>(zenoh::open(std::move(config)));

    char z_topic_name[kMaxTopicNameLen + 6];
    sprintf(z_topic_name, "shoz/%s", topic_name);
    this->z_publisher = zenoh::expect<zenoh::Publisher>(z_session.declare_publisher(z_topic_name));

    switch (domain.dev_type) {
    case DeviceType::kGPU:
        this->allocator = (Allocator*) new ShareableAllocator((TopicHeader*) this->shm_base);
        break;
    }
}

void Publisher::put(void* payload, size_t size) {
    size_t offset = this->allocator->malloc(size);
    if (offset == -1) throwError("No free space in the pool");
    void* addr = (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);
    this->allocator->copyTo(addr, payload, size);

    // get next available index as message id
    MessageQueueHeader* mq_header = getMessageQueueHeader(getDomainMap(getTlsfHeader(getTopicHeader(this->shm_base))));
    size_t msg_id = std::atomic_ref<size_t>(mq_header->next).fetch_add(1) % kMaxMessageNum;
    MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header, msg_id);

    // TODO: free the the corresponding space if it hasn't been freed

    // NOTE: though `taken_num` and `avail` should be atomic referenced, it's okay because
    //       no other process will access these variables at this moment
    mq_entry->taken_num = 0;
    mq_entry->offset = offset;
    mq_entry->avail = 1 << this->domain_idx;

    // notify subscribers with the message ID
    zenoh::BytesView msg((void*) &msg_id, sizeof(size_t));
    if (!this->z_publisher.put(msg)) {
        throwError("Warning: Zenoh failed to send a message");
    }
}