#ifndef PUBLISHER_HPP
#define PUBLISHER_HPP

#include <cstddef>
#include <string>

#include <zenoh.hxx>

#include "metadata.hpp"
#include "node/node.hpp"

#ifdef BUILD_PYGAPS
#include <nanobind/ndarray.h>

#include "pygaps.hpp"
#include "zenoh_wrapper.hpp"

namespace nb = nanobind;

typedef nb::ndarray<nb::pytorch, nb::device::cuda> DeviceTensor;
typedef ZenohSession session_t;
#else
typedef zenoh::Session session_t;
#endif

class Publisher : public Node {
public:
    Publisher() = delete;
    Publisher(const session_t& session, std::string&& topic_name,
              size_t pool_size, int msg_queue_cap_exp);

#ifdef BUILD_PYGAPS
    DeviceTensor empty(nb::tuple shape, Dtype dtype);
    void put(const DeviceTensor& tensor);
#else
    void* malloc(size_t size);
    void put(void* payload, size_t size);
#endif

protected:
    inline void updateEntry(MessageQueueEntry* mq_entry, size_t offset,
                            size_t size) const {
        // free the payload's space if it hasn't been freed
        // NOTE: since `offset` from the allocator is always even,
        // we use its first bit to determine if the space is freed
        if (mq_entry->offset & 1) {
            this->allocator->free(mq_entry->offset);
        }

        // NOTE: though `taken_num` should be atomic referenced, it's
        // okay because no other process will access it at this moment
        mq_entry->taken_num = 0;
        mq_entry->offset = offset | 1;
        mq_entry->size = size;
    }

    zenoh::Publisher z_publisher;
};

#endif  // PUBLISHER_HPP