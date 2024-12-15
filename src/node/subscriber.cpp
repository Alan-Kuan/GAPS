#include "node/subscriber.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include <zenoh-pico/config.h>
#include <zenoh.hxx>

#include "allocator.hpp"
#include "metadata.hpp"

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>
namespace nb = nanobind;
#endif

Subscriber::Subscriber(const zenoh::Session& z_session,
                       std::string&& topic_name, size_t pool_size,
                       MessageHandler handler)
        : Node(topic_name.c_str(), pool_size),
          z_subscriber(z_session.declare_subscriber("shoz/" + topic_name,
                                                    this->makeCallback(handler),
                                                    zenoh::closures::none)) {
    this->allocator = new Allocator((TopicHeader*) this->shm_base, true);
}

Subscriber::~Subscriber() {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    std::atomic_ref<uint32_t>(topic_header->sub_count)--;
}

std::function<void(const zenoh::Sample&)> Subscriber::makeCallback(
    MessageHandler handler) {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    std::atomic_ref<uint32_t>(topic_header->sub_count)++;

    MessageQueueHeader* mq_header =
        getMessageQueueHeader(getTlsfHeader(topic_header));

    auto callback = [=, this](const zenoh::Sample& sample) {
#ifdef BUILD_PYSHOZ
        MsgBuf msg_buf;
        memcpy(&msg_buf, sample.get_payload().as_vector().data(),
               sizeof(MsgBuf));
        MessageQueueEntry* mq_entry =
            getMessageQueueEntry(mq_header, msg_buf->msg_id);
#else
        size_t msg_id;
        memcpy(&msg_id, sample.get_payload().as_vector().data(),
               sizeof(msg_id));
        MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header, msg_id);
#endif

        size_t offset = mq_entry->offset ^ 1;
        void* data =
            (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);

#ifdef BUILD_PYSHOZ
        {
            nb::gil_scoped_acquire acq;
            const int64_t* strides =
                msg_buf->strides[0] ? msg_buf->strides : nullptr;
            DeviceTensor tensor(
                data, msg_buf->ndim, (const size_t*) msg_buf->shape,
                nb::handle(), strides, msg_buf->dtype, nb::device::cuda::value);
            handler(tensor);
        }
#else
        handler(data, mq_entry->size);
#endif

        // last subscriber reading the message should free the allocation
        if (std::atomic_ref<uint32_t>(mq_entry->taken_num).fetch_add(1) ==
            topic_header->sub_count - 1) {
            this->allocator->free(offset);
            // remove the label that indicates the payload is not freed
            mq_entry->offset = offset;
        }
    };
    return callback;
}