#include "node/publisher.hpp"

#include <alloca.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "allocator/shareable.hpp"

Publisher::Publisher(const char* topic_name, const char* conf_path, const Allocator::Domain& domain, size_t pool_size)
        : Node(topic_name),
          z_session(nullptr),
          z_publisher(nullptr),
          domain_id(domain.getId()) {
    MessageQueueHeader* mqh = (MessageQueueHeader*) ((uint8_t*) this->shm_base + sizeof(Allocator::Metadata));
    mqh->capacity = kMaxMessageNum;
    mqh->next = 0;
    mqh->sub_count = 0;

    auto config = zenoh::expect<zenoh::Config>(zenoh::config_from_file(conf_path));
    this->z_session = zenoh::expect<zenoh::Session>(zenoh::open(std::move(config)));

    char z_topic_name[kMaxTopicNameLen + 6];
    sprintf(z_topic_name, "shoz/%s", topic_name);
    this->z_publisher = zenoh::expect<zenoh::Publisher>(z_session.declare_publisher(z_topic_name));

    switch (domain.dev_type) {
    case Allocator::DeviceType::kGPU:
        this->allocator = (Allocator*) new ShareableAllocator((Allocator::Metadata*) this->shm_base, pool_size);
        // TODO: make it more flexible
        ((ShareableAllocator*) this->allocator)->shareHandle(1);
        break;
    }
}

Publisher::~Publisher(void) {
    delete this->allocator;
}

void Publisher::put(void* payload, size_t size) {
    // get next available index as message id
    MessageQueueHeader* mqh = (MessageQueueHeader*) ((uint8_t*) this->shm_base + sizeof(Allocator::Metadata) + sizeof(MessageQueueHeader));
    size_t msg_id = std::atomic_ref<size_t>(mqh->next).fetch_add(1) % kMaxMessageNum;
    uint8_t* msg_entry = (uint8_t*) mqh + msg_id * (sizeof(int) + kMaxDomainNum * sizeof(size_t));

    std::atomic_ref<int> untaken_num{*((int*) msg_entry)};
    untaken_num = mqh->sub_count;

    // go through each subscribed domain and allocate a space and put the data there
    void* addr = this->allocator->malloc(size);
    size_t offset = (uint8_t*) addr - (uint8_t*) this->allocator->getPoolBase();
    size_t* offsets = (size_t*) (msg_entry + sizeof(int));
    offsets[this->domain_id] = offset;

    this->allocator->copyTo(addr, payload, size);

    zenoh::BytesView msg((void*) &msg_id, sizeof(size_t));
    if (!this->z_publisher.put(msg)) {
        std::cerr << "Warning: Zenoh failed to send a message" << std::endl;
    }
}