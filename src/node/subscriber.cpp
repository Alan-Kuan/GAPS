#include "node/subscriber.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "allocator/shareable.hpp"
#include "error.hpp"

Subscriber::Subscriber(const char* topic_name, const char* conf_path, const Allocator::Domain& domain)
        : Node(topic_name),
          z_session(nullptr),
          z_subscriber(nullptr),
          domain_id(domain.getId()) {
    auto config = zenoh::expect<zenoh::Config>(zenoh::config_from_file(conf_path));
    this->z_session = zenoh::expect<zenoh::Session>(zenoh::open(std::move(config)));

    switch (domain.dev_type) {
    case Allocator::DeviceType::kGPU:
        this->allocator = (Allocator*) new ShareableAllocator((Allocator::Metadata*) this->shm_base);
        // TODO: make it more flexible
        ((ShareableAllocator*) this->allocator)->recvHandle();
        break;
    }
}

Subscriber::~Subscriber() {
    delete this->allocator;
}

void Subscriber::sub(MessageHandler handler) {
    MessageQueueHeader* mqh = (MessageQueueHeader*) ((uint8_t*) this->shm_base + sizeof(Allocator::Metadata));
    std::atomic_ref<uint32_t>(mqh->sub_count)++;

    auto callback = [handler, mqh, this](const zenoh::Sample& sample) {
        zenoh::BytesView msg = sample.get_payload();
        size_t msg_id = *((size_t*) msg.as_string_view().data());

        uint8_t* msg_entry = (uint8_t*) mqh + sizeof(MessageQueueHeader) + msg_id * (sizeof(int) + kMaxDomainNum * sizeof(size_t));
        size_t* offsets = (size_t*) (msg_entry + sizeof(int));
        uint8_t* data = (uint8_t*) this->allocator->getPoolBase() + offsets[this->domain_id];

        // TODO: check if data is valid; if not, copy valid ones to this domain

        std::atomic_ref<int> untaken_num = std::atomic_ref<int>(*((int*) msg_entry));
        if (untaken_num.fetch_sub(1) == 1) {
            // TODO: free the memory via corresponding allocator
        }

        handler((void*) data);
    };

    try {
        char z_topic_name[kMaxTopicNameLen + 6];
        char* topic_name = (char*) this->shm_base;
        sprintf(z_topic_name, "shoz/%s", topic_name);
        this->z_subscriber = zenoh::expect<zenoh::Subscriber>(this->z_session.declare_subscriber(z_topic_name, std::move(callback)));
    } catch (zenoh::ErrorMessage& err) {
        throwError(err.as_string_view().data());
    }
}