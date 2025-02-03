#include "node/subscriber.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>

#include <iceoryx_posh/capro/service_description.hpp>
#include <iceoryx_posh/popo/subscriber.hpp>

#include "allocator/tlsf.hpp"
#include "error.hpp"
#include "metadata.hpp"

#ifdef BUILD_PYSHOI
#include <nanobind/ndarray.h>

namespace nb = nanobind;
#endif

Subscriber::Subscriber(const char* topic_name, size_t pool_size,
                       int msg_queue_cap_exp, MessageHandler handler)
        : Node(topic_name, pool_size, msg_queue_cap_exp),
          iox_subscriber({"", "shoi",
                          iox::capro::IdString_t(iox::cxx::TruncateToCapacity,
                                                 topic_name)},
                         {.queueCapacity = 256}),
          handler(handler) {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    std::atomic_ref<uint32_t>(topic_header->sub_count)++;
    this->mq_header = getMessageQueueHeader(getTlsfHeader(topic_header));

    this->allocator = new TlsfAllocator((TopicHeader*) this->shm_base, true);

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

void Subscriber::onSampleReceived(iox_subscriber_t* iox_subscriber,
                                  Subscriber* self) {
    bool keep = true;
    for (int i = 0; i < 5 && keep; i++) {
        iox_subscriber
            ->take()
        // TODO
#ifdef BUILD_PYSHOI
            .and_then([iox_subscriber, self](const void* payload) {
                auto msg_header = (MsgHeader*) payload;
                auto shape_buf =
                    (size_t*) ((uintptr_t) msg_header + sizeof(MsgHeader));
                MessageQueueEntry* mq_entry =
                    getMessageQueueEntry(self->mq_header, msg_header->msg_id);
#else
            .and_then([iox_subscriber, self](auto& msg_id) {
                MessageQueueEntry* mq_entry =
                    getMessageQueueEntry(self->mq_header, *msg_id);
#endif
                TopicHeader* topic_header = getTopicHeader(self->shm_base);

                size_t offset = mq_entry->offset ^ 1;
                void* data =
                    (void*) ((uintptr_t) self->allocator->getPoolBase() +
                             offset);

#ifdef BUILD_PYSHOI
                {
                    nb::gil_scoped_acquire acq;
                    DeviceTensor tensor(
                        data, msg_header->ndim, shape_buf, nb::handle(),
                        nullptr, msg_header->dtype, nb::device::cuda::value);
                    self->handler(tensor);
                }
#else
                self->handler(data, mq_entry->size);
#endif

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