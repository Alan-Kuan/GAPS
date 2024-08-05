#include "node/python_iface/publisher.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <zenoh.hxx>

#include "error.hpp"
#include "metadata.hpp"
#include "node/publisher.hpp"

Publisher::Publisher(const char* topic_name, const char* llocator,
                     const Domain& domain, size_t pool_size)
        : __Publisher(topic_name, llocator, domain, pool_size) {}

void Publisher::put(void* payload, size_t size) {
    // if (!payload) throwError("Payload was not provided");
    // if (size == 0) return;

    // size_t offset = this->allocator->malloc(size);
    // if (offset == -1) throwError("No free space in the pool");
    // void* addr = (void*) ((uintptr_t) this->allocator->getPoolBase() +
    // offset); this->allocator->copyTo(addr, payload, size);

    // // get next available index as message id
    // MessageQueueHeader* mq_header = getMessageQueueHeader(
    //     getDomainMap(getTlsfHeader(getTopicHeader(this->shm_base))));
    // size_t msg_id =
    //     std::atomic_ref<size_t>(mq_header->next).fetch_add(1) %
    //     kMaxMessageNum;
    // MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header, msg_id);

    // // free the payload's space if it hasn't been freed
    // if (mq_entry->avail) {
    //     this->allocator->free(mq_entry->offset);
    // }

    // // NOTE: though `taken_num` and `avail` should be atomic referenced, it's
    // // okay because no other process will access these variables at this
    // moment mq_entry->taken_num = 0; mq_entry->offset = offset; mq_entry->size
    // = size; mq_entry->avail = 1 << this->domain_idx;

    // // notify subscribers with the message ID
    // zenoh::BytesView msg((void*) &msg_id, sizeof(size_t));
    // if (!this->z_publisher.put(msg)) {
    //     throwError("Zenoh failed to send a message");
    // }
}