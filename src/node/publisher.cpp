#include "node/publisher.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <zenoh-pico/config.h>
#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "allocator/shareable.hpp"
#include "error.hpp"
#include "metadata.hpp"

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>

namespace nb = nanobind;

struct MsgBuf {
    size_t msg_id;
    nb::dlpack::dtype dtype;
    int32_t ndim;
    int64_t shape[3];
    int64_t strides[3];
};
#endif

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

#ifdef BUILD_PYSHOZ
void Publisher::put(const nb::ndarray<>& tensor) {
    if (tensor.ndim() > 3) {
        throwError("Tensor with dimension greater than 3 is not supported");
    }

    size_t size = tensor.nbytes();
    if (size == 0) return;

    MsgBuf msg_buf;
    msg_buf.dtype = tensor.dtype();
    msg_buf.ndim = tensor.ndim();
    for (int i = 0; i < msg_buf.ndim; i++) {
        msg_buf.shape[i] = tensor.shape(i);
    }
    if (tensor.stride_ptr()) {
        for (int i = 0; i < msg_buf.ndim; i++) {
            msg_buf.strides[i] = tensor.stride(i);
        }
    } else {
        msg_buf.strides[0] = 0;
    }
#else
void Publisher::put(void* payload, size_t size) {
    if (!payload) throwError("Payload was not provided");
    if (size == 0) return;
#endif

    size_t offset = this->allocator->malloc(size);
    if (offset == -1) throwError("No free space in the pool");
    void* addr = (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);

#ifdef BUILD_PYSHOZ
    this->allocator->copyTo(addr, (void*) tensor.data(), size);
#else
    this->allocator->copyTo(addr, payload, size);
#endif

    // get next available index as message id
    MessageQueueHeader* mq_header = getMessageQueueHeader(
        getDomainMap(getTlsfHeader(getTopicHeader(this->shm_base))));
#ifdef BUILD_PYSHOZ
    msg_buf.msg_id =
        std::atomic_ref<size_t>(mq_header->next).fetch_add(1) % kMaxMessageNum;
    MessageQueueEntry* mq_entry =
        getMessageQueueEntry(mq_header, msg_buf.msg_id);
#else
    size_t msg_id =
        std::atomic_ref<size_t>(mq_header->next).fetch_add(1) % kMaxMessageNum;
    MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header, msg_id);
#endif

    // free the payload's space if it hasn't been freed
    if (mq_entry->avail) {
        this->allocator->free(mq_entry->offset);
    }

    // NOTE: though `taken_num` and `avail` should be atomic referenced, it's
    // okay because no other process will access these variables at this moment
    mq_entry->taken_num = 0;
    mq_entry->offset = offset;
    mq_entry->size = size;
    mq_entry->avail = 1 << this->domain_idx;

#ifdef BUILD_PYSHOZ
    // notify subscribers with the message ID & tensor info
    zenoh::BytesView msg((void*) &msg_buf, sizeof(msg_buf));
#else
    // notify subscribers with the message ID
    zenoh::BytesView msg((void*) &msg_id, sizeof(size_t));
#endif
    if (!this->z_publisher.put(msg)) {
        throwError("Zenoh failed to send a message");
    }
}