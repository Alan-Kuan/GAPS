#include "node/cpp_iface/subscriber.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <zenoh.hxx>

#include "error.hpp"
#include "metadata.hpp"
#include "node/subscriber.hpp"

Subscriber::Subscriber(const char* topic_name, const char* llocator,
                       const Domain& domain, size_t pool_size)
        : __Subscriber(topic_name, llocator, domain, pool_size) {}

void Subscriber::sub(MessageHandler handler) {
    // TopicHeader* topic_header = getTopicHeader(this->shm_base);
    // std::atomic_ref<uint32_t>(topic_header->sub_count)++;

    // MessageQueueHeader* mq_header =
    //     getMessageQueueHeader(getDomainMap(getTlsfHeader(topic_header)));

    // auto callback = [=, this](const zenoh::Sample& sample) {
    //     zenoh::BytesView msg = sample.get_payload();
    //     size_t msg_id = *((size_t*) msg.as_string_view().data());

    //     MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header,
    //     msg_id); size_t offset = mq_entry->offset;
    //     // TODO: check availability; if not available, copy valid ones to
    //     this
    //     // domain
    //     void* data =
    //         (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);

    //     handler(data, mq_entry->size);

    //     // last subscriber reading the message should free the allocation
    //     if (std::atomic_ref<uint32_t>(mq_entry->taken_num).fetch_add(1) ==
    //         topic_header->sub_count - 1) {
    //         this->allocator->free(offset);
    //         mq_entry->avail = 0;
    //     }
    // };

    // try {
    //     char z_topic_name[kMaxTopicNameLen + 6];
    //     char* topic_name = (char*) this->shm_base;
    //     sprintf(z_topic_name, "shoz/%s", topic_name);
    //     this->z_subscriber =
    //         zenoh::expect<zenoh::Subscriber>(this->z_session.declare_subscriber(
    //             z_topic_name, std::move(callback)));
    // } catch (zenoh::ErrorMessage& err) {
    //     throwError(err.as_string_view().data());
    // }
}