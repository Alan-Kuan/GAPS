#include "node/publisher.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <zenoh-pico/config.h>
#include <zenoh.hxx>

#include "allocator/tlsf.hpp"
#include "error.hpp"
#include "metadata.hpp"

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>

#include "pyshoz.hpp"
#include "zenoh_wrapper.hpp"

namespace nb = nanobind;
#endif

Publisher::Publisher(const session_t& session, std::string&& topic_name,
                     size_t pool_size, int msg_queue_cap_exp)
        : Node(topic_name.c_str(), pool_size, msg_queue_cap_exp),
#ifdef BUILD_PYSHOZ
          z_publisher(
              session.getSession().declare_publisher("shoz/" + topic_name,
#else
          z_publisher(session.declare_publisher("shoz/" + topic_name,
#endif
                                                     {.is_express = true})) {
    this->allocator = new TlsfAllocator((TopicHeader*) this->shm_base);
}

#ifdef BUILD_PYSHOZ
DeviceTensor Publisher::empty(nb::tuple shape, Dtype dtype) {
    nb::dlpack::dtype nb_dtype;
    switch (dtype) {
    case Dtype::int8:
        nb_dtype = {.code = 0, .bits = 8, .lanes = 1};
        break;
    case Dtype::int16:
        nb_dtype = {.code = 0, .bits = 16, .lanes = 1};
        break;
    case Dtype::int32:
        nb_dtype = {.code = 0, .bits = 32, .lanes = 1};
        break;
    case Dtype::int64:
        nb_dtype = {.code = 0, .bits = 64, .lanes = 1};
        break;
    case Dtype::uint8:
        nb_dtype = {.code = 1, .bits = 8, .lanes = 1};
        break;
    case Dtype::float16:
        nb_dtype = {.code = 2, .bits = 16, .lanes = 1};
        break;
    case Dtype::float32:
        nb_dtype = {.code = 2, .bits = 32, .lanes = 1};
        break;
    }

    int32_t ndim = shape.size();
    std::vector<size_t> shape_buf;
    size_t size = 1;
    for (auto it = shape.begin(); it != shape.end(); it++) {
        size_t val = nb::cast<size_t>(*it);
        size *= val;
        shape_buf.push_back(val);
    }
    size *= (nb_dtype.bits * nb_dtype.lanes + 7) / 8;

    size_t offset = this->allocator->malloc(size);
    if (offset == -1) throwError("no available space");
    auto addr = (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);

    return DeviceTensor(addr, ndim, shape_buf.data(), nb::handle(), nullptr,
                        nb_dtype, nb::device::cuda::value);
}
#else
void* Publisher::malloc(size_t size) {
    size_t offset = this->allocator->malloc(size);
    if (offset == -1) return nullptr;
    return (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);
}
#endif

// Some parts of Publisher::put is common between C++ version and Python
// version, so it is written as below.
#ifdef BUILD_PYSHOZ
void Publisher::put(const DeviceTensor& tensor) {
    size_t size = tensor.nbytes();
    if (size == 0) return;

    MsgHeader msg_header;
    msg_header.dtype = tensor.dtype();
    msg_header.ndim = tensor.ndim();

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
    msg_header.msg_id = std::atomic_ref<size_t>(mq_header->next).fetch_add(1) %
                        mq_header->capacity;
    MessageQueueEntry* mq_entry =
        getMessageQueueEntry(mq_header, msg_header.msg_id);
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
    std::vector<uint8_t> byte_arr(sizeof(msg_header) +
                                  sizeof(int64_t) * msg_header.ndim);
    auto shape_buf =
        (size_t*) ((uintptr_t) byte_arr.data() + sizeof(msg_header));
    memcpy(byte_arr.data(), &msg_header, sizeof(msg_header));
    for (int i = 0; i < msg_header.ndim; i++) {
        shape_buf[i] = tensor.shape(i);
    }
#else
    // notify subscribers with the message ID
    std::vector<uint8_t> byte_arr(sizeof(msg_id));
    memcpy(byte_arr.data(), &msg_id, sizeof(msg_id));
#endif
    this->z_publisher.put(zenoh::Bytes(std::move(byte_arr)));
}