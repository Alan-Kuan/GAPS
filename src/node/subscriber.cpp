#include "node/subscriber.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <iceoryx_posh/capro/service_description.hpp>
#include <iceoryx_posh/popo/subscriber.hpp>

#include "allocator.hpp"
#include "error.hpp"
#include "metadata.hpp"

Subscriber::Subscriber(const char* topic_name, size_t pool_size,
                       MessageHandler handler)
        : Node(topic_name, pool_size),
          iox_subscriber({"", "shoi",
                          iox::capro::IdString_t(iox::cxx::TruncateToCapacity,
                                                 topic_name)},
                         {.queueCapacity = 256}),
          handler(handler) {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    std::atomic_ref<uint32_t>(topic_header->sub_count)++;
    this->mq_header = getMessageQueueHeader(getTlsfHeader(topic_header));

    this->allocator = new Allocator((TopicHeader*) this->shm_base, true);

    this->iox_listener
        .attachEvent(this->iox_subscriber,
                     iox::popo::SubscriberEvent::DATA_RECEIVED,
                     iox::popo::createNotificationCallback(
                         this->onSampleReceived, *this))
        .or_else(
            [](auto) { throwError("Failed to register message handler"); });
}

Subscriber::~Subscriber() {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    std::atomic_ref<uint32_t>(topic_header->sub_count)--;
}

void Subscriber::onSampleReceived(iox::popo::Subscriber<size_t>* iox_subscriber,
                                  Subscriber* self) {
    bool keep = true;
    for (int i = 0; i < 10 && keep; i++) {
        iox_subscriber->take()
            .and_then([iox_subscriber, self](auto& msg_id) {
                TopicHeader* topic_header = getTopicHeader(self->shm_base);
                MessageQueueEntry* mq_entry =
                    getMessageQueueEntry(self->mq_header, *msg_id);

                size_t offset = mq_entry->offset ^ 1;
                void* data =
                    (void*) ((uintptr_t) self->allocator->getPoolBase() +
                             offset);

                self->handler(data, mq_entry->size);

                // last subscriber reading the message should free the
                // allocation
                if (std::atomic_ref<uint32_t>(mq_entry->taken_num)
                        .fetch_add(1) == topic_header->sub_count - 1) {
                    self->allocator->free(offset);
                    // remove the label that indicates the payload is not freed
                    mq_entry->offset = offset;
                }
            })
            .or_else([&keep](auto& result) { keep = false; });
    }
}