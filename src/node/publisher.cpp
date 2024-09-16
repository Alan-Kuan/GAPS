#include "node/publisher.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>
#include <zenoh-pico/config.h>
#include <zenoh.hxx>

#include "allocator.hpp"
#include "error.hpp"
#include "metadata.hpp"

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>
namespace nb = nanobind;
#endif

Publisher::Publisher(const char* topic_name, const char* llocator,
                     size_t pool_size)
        : Node(topic_name, pool_size),
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

    this->allocator = new Allocator((TopicHeader*) this->shm_base);
}

#ifdef BUILD_PYSHOZ
void Publisher::copyTensor(DeviceTensor& dst,
                           const nb::ndarray<nb::pytorch>& src) {
    auto kind = src.device_type() == nb::device::cpu::value
                    ? cudaMemcpyHostToDevice
                    : cudaMemcpyDeviceToDevice;
    cudaMemcpy(dst.data(), src.data(), src.nbytes(), kind);
}

DeviceTensor Publisher::malloc(size_t ndim, nb::tuple shape,
                               nb::tuple dtype_tup, bool clean) {
    if (ndim != shape.size()) {
        throwError("'ndim' does not match the size of 'shape'");
    }
    if (ndim > 3) {
        throwError("Currently supports only at most 3 dimensions");
    }

    uint8_t code = nb::cast<uint8_t>(dtype_tup[0]);
    uint8_t bits = nb::cast<uint8_t>(dtype_tup[1]);
    uint16_t lanes = nb::cast<uint16_t>(dtype_tup[2]);
    nb::dlpack::dtype dtype{.code = code, .bits = bits, .lanes = lanes};

    size_t shape_buf[3];
    size_t size = (bits / 8) * ndim;
    int i = 0;
    for (auto it = shape.begin(); it != shape.end(); it++) {
        size_t val = nb::cast<size_t>(*it);
        size *= val;
        shape_buf[i++] = val;
    }

    size_t offset = this->allocator->malloc(size);
    if (offset == -1) throwError("no available space");
    auto addr = (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);

    if (clean) cudaMemset(addr, 0, size);

    return DeviceTensor(addr, ndim, shape_buf, nb::handle(), nullptr, dtype,
                        nb::device::cuda::value);
}
#else
void* Publisher::malloc(size_t size) {
    size_t offset = this->allocator->malloc(size);
    if (offset == -1) return nullptr;
    return (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);
}
#endif

#ifdef BUILD_PYSHOZ
void Publisher::put(const DeviceTensor& tensor) {
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

    size_t offset =
        (uintptr_t) tensor.data() - (uintptr_t) this->allocator->getPoolBase();
#else
void Publisher::put(void* payload, size_t size) {
    if (!payload) throwError("Payload was not provided");
    if (size == 0) return;
    size_t offset =
        (uintptr_t) payload - (uintptr_t) this->allocator->getPoolBase();
#endif
    // get next available index as message id
    MessageQueueHeader* mq_header =
        getMessageQueueHeader(getTlsfHeader(getTopicHeader(this->shm_base)));
#ifdef BUILD_PYSHOZ
    msg_buf.msg_id = std::atomic_ref<size_t>(mq_header->next).fetch_add(1) %
                     mq_header->capacity;
    MessageQueueEntry* mq_entry =
        getMessageQueueEntry(mq_header, msg_buf.msg_id);
#else
    size_t msg_id = std::atomic_ref<size_t>(mq_header->next).fetch_add(1) %
                    mq_header->capacity;
    MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header, msg_id);
#endif

    // free the payload's space if it hasn't been freed
    // NOTE: since `offset` from the allocator is always even,
    // we use its first bit to determine if the space is freed
    if (mq_entry->offset & 1) {
        this->allocator->free(mq_entry->offset);
    }

    // NOTE: though `taken_num` and `avail` should be atomic referenced, it's
    // okay because no other process will access these variables at this moment
    mq_entry->taken_num = 0;
    mq_entry->offset = offset | 1;
    mq_entry->size = size;

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