#include "node/subscriber.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>

#include <iceoryx_posh/capro/service_description.hpp>
#include <iceoryx_posh/popo/subscriber.hpp>

#include "error.hpp"
#include "metadata.hpp"
#include "profiling.hpp"

#ifdef BUILD_PYGAPS
#include <nanobind/ndarray.h>

namespace nb = nanobind;
#endif

Subscriber::Subscriber(const char* topic_name, size_t pool_size,
                       int msg_queue_cap_exp, MessageHandler handler)
        : Node(topic_name, pool_size, msg_queue_cap_exp, true),
          iox_subscriber({"", "gaps",
                          iox::capro::IdString_t(iox::cxx::TruncateToCapacity,
                                                 topic_name)},
                         {.queueCapacity = 256}),
          handler(handler) {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    std::atomic_ref<uint32_t>(topic_header->sub_count)++;
    this->mq_header = getMessageQueueHeader(getTlsfHeader(topic_header));

    this->iox_listener
        .attachEvent(this->iox_subscriber,
                     iox::popo::SubscriberEvent::DATA_RECEIVED,
                     iox::popo::createNotificationCallback(
                         this->onSampleReceived, *this))
        .or_else(
            [](auto) { throwError("Failed to register message handler"); });
    PROF_WARN;
}

Subscriber::~Subscriber() {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    std::atomic_ref<uint32_t>(topic_header->sub_count)--;
}

void Subscriber::onSampleReceived(iox_subscriber_t* iox_subscriber,
                                  Subscriber* self) {
    bool keep = true;
    while (keep) {
        PROF_ADD_POINT;

        iox_subscriber
            ->take()
#ifdef BUILD_PYGAPS
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
                PROF_ADD_POINT;

#ifdef BUILD_PYGAPS
                {
                    nb::gil_scoped_acquire acq;
                    // NOTE: use nb::bytes("0") to trick nanobind into
                    // believing it's the owner; otherwise, nanobind will
                    // copy the tensor (since nanobind 2.2.0), which is what
                    // we want to avoid
                    DeviceTensor tensor(
                        data, msg_header->ndim, shape_buf, nb::bytes("0"),
                        nullptr, msg_header->dtype, nb::device::cuda::value);
                    self->handler(tensor);
                }
#else
                self->handler(data, mq_entry->size);
#endif
                PROF_ADD_POINT;

                // last subscriber reading the message should free the
                // allocation
                if (std::atomic_ref<uint32_t>(mq_entry->taken_num)
                        .fetch_add(1) == topic_header->sub_count - 1) {
                    self->allocator->free(offset);
                    // remove the label that indicates the payload is not
                    // freed
                    mq_entry->offset = offset;
                }
                PROF_ADD_POINT;

#if BUILD_PYGAPS
                PROF_ADD_TAG(msg_header->msg_id);
#else
                PROF_ADD_TAG(*msg_id);
#endif
            })
            .or_else([&keep](auto& result) { keep = false; });
    }
}
