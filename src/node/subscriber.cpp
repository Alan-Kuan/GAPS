#include "node/subscriber.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <zenoh-pico/config.h>
#include <zenoh.hxx>

#include "metadata.hpp"
#include "profiling.hpp"

#ifdef BUILD_PYGAPS
#include <nanobind/ndarray.h>

#include "zenoh_wrapper.hpp"

namespace nb = nanobind;
#endif

Subscriber::Subscriber(const session_t& session, std::string&& topic_name,
                       size_t pool_size, int msg_queue_cap_exp,
                       MessageHandler handler)
        : Node(topic_name.c_str(), pool_size, msg_queue_cap_exp, true),
#ifdef BUILD_PYGAPS
          z_subscriber(session.getSession().declare_subscriber(
              "gaps/" + topic_name,
#else
          z_subscriber(session.declare_subscriber(
              "gaps/" + topic_name,
#endif
              this->makeCallback(handler), zenoh::closures::none)) {
    PROF_WARN;
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
        PROF_ADD_POINT;

#ifdef BUILD_PYGAPS
        std::vector<uint8_t> raw_msg{sample.get_payload().as_vector()};
        auto msg_header = (MsgHeader*) raw_msg.data();
        auto shape_buf = (size_t*) ((uintptr_t) msg_header + sizeof(MsgHeader));
        MessageQueueEntry* mq_entry =
            getMessageQueueEntry(mq_header, msg_header->msg_id);
#else
        size_t msg_id;
        memcpy(&msg_id, sample.get_payload().as_vector().data(),
               sizeof(msg_id));
        MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header, msg_id);
#endif

        size_t offset = mq_entry->offset ^ 1;
        void* data =
            (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);
        PROF_ADD_POINT;

#ifdef BUILD_PYGAPS
        {
            nb::gil_scoped_acquire acq;
            // NOTE: use nb::bytes("0") to trick nanobind into believing it's
            // the owner; otherwise, nanobind will copy the tensor (since
            // nanobind 2.2.0), which is what we want to avoid
            DeviceTensor tensor(data, msg_header->ndim, shape_buf,
                                nb::bytes("0"), nullptr, msg_header->dtype,
                                nb::device::cuda::value);
            handler(tensor);
        }
#else
        handler(data, mq_entry->size);
#endif
        PROF_ADD_POINT;

        // last subscriber reading the message should free the allocation
        if (std::atomic_ref<uint32_t>(mq_entry->taken_num).fetch_add(1) ==
            topic_header->sub_count - 1) {
            this->allocator->free(offset);
            // remove the label that indicates the payload is not freed
            mq_entry->offset = offset;
        }
        PROF_ADD_POINT;

#ifdef BUILD_PYGAPS
        PROF_ADD_TAG(msg_header->msg_id);
#else
        PROF_ADD_TAG(msg_id);
#endif
    };
    return callback;
}