#include "node/publisher.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>

#include <iceoryx_posh/capro/service_description.hpp>
#include <iceoryx_posh/popo/publisher.hpp>

#include "error.hpp"
#include "metadata.hpp"
#include "profiling.hpp"

#ifdef BUILD_PYSHOI
#include <cuda_runtime.h>
#include <nanobind/ndarray.h>

#include "pyshoi.hpp"

namespace nb = nanobind;
#endif

Publisher::Publisher(const char* topic_name, size_t pool_size,
                     int msg_queue_cap_exp)
        : Node(topic_name, pool_size, msg_queue_cap_exp),
          iox_publisher({"", "shoi",
                         iox::capro::IdString_t(iox::cxx::TruncateToCapacity,
                                                topic_name)}) {
    PROFILE_WARN;
}

#ifdef BUILD_PYSHOI
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

void Publisher::put(const DeviceTensor& tensor) {
    PROFILE_INIT(3);
    PROFILE_SETPOINT(0);

    size_t size = tensor.nbytes();
    if (size == 0) return;

    MsgHeader msg_header;
    msg_header.dtype = tensor.dtype();
    msg_header.ndim = tensor.ndim();

    size_t offset =
        (uintptr_t) tensor.data() - (uintptr_t) this->allocator->getPoolBase();

    // get next available index as message id
    MessageQueueHeader* mq_header =
        getMessageQueueHeader(getTlsfHeader(getTopicHeader(this->shm_base)));
    msg_header.msg_id = std::atomic_ref<size_t>(mq_header->next).fetch_add(1) %
                        mq_header->capacity;
    MessageQueueEntry* mq_entry =
        getMessageQueueEntry(mq_header, msg_header.msg_id);

    this->updateEntry(mq_entry, offset, size);
    PROFILE_SETPOINT(1);

    // notify subscribers with the message ID & tensor info
    this->iox_publisher
        .loan(sizeof(msg_header) + sizeof(int64_t) * msg_header.ndim)
        .and_then([this, &msg_header, &tensor](void* buf) {
            auto shape_buf = (size_t*) ((uintptr_t) buf + sizeof(msg_header));
            memcpy(buf, &msg_header, sizeof(msg_header));
            for (int i = 0; i < msg_header.ndim; i++) {
                shape_buf[i] = tensor.shape(i);
            }
            this->iox_publisher.publish(buf);
        })
        .or_else([](auto& error) {
            std::stringstream ss;
            ss << "Iceoryx failed to send a message: " << error;
            throwError(ss.str().c_str());
        });
    PROFILE_SETPOINT(2);

    PROFILE_OUTPUT(3);
}
#else
void* Publisher::malloc(size_t size) {
    size_t offset = this->allocator->malloc(size);
    if (offset == -1) return nullptr;
    return (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);
}

void Publisher::put(void* payload, size_t size) {
    PROFILE_INIT(3);
    PROFILE_SETPOINT(0);

    if (!payload) throwError("Payload was not provided");
    if (size == 0) return;

    size_t offset =
        (uintptr_t) payload - (uintptr_t) this->allocator->getPoolBase();

    // get next available index as message id
    MessageQueueHeader* mq_header =
        getMessageQueueHeader(getTlsfHeader(getTopicHeader(this->shm_base)));
    size_t msg_id = std::atomic_ref<size_t>(mq_header->next).fetch_add(1) %
                    mq_header->capacity;
    MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header, msg_id);

    this->updateEntry(mq_entry, offset, size);
    PROFILE_SETPOINT(1);

    // notify subscribers with the message ID
    this->iox_publisher.publishCopyOf(msg_id).or_else([](auto& error) {
        std::stringstream ss;
        ss << "Iceoryx failed to send a message: " << error;
        throwError(ss.str().c_str());
    });
    PROFILE_SETPOINT(2);

    PROFILE_OUTPUT(3);
}
#endif
