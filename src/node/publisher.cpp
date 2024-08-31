#include "node/publisher.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>

#include <iceoryx_posh/capro/service_description.hpp>
#include <iceoryx_posh/popo/publisher.hpp>

#include "allocator/allocator.hpp"
#include "allocator/shareable.hpp"
#include "error.hpp"
#include "metadata.hpp"

Publisher::Publisher(const char* topic_name, size_t pool_size)
        : Node(topic_name, pool_size),
          iox_publisher({"", "shoi",
                         iox::capro::IdString_t(iox::cxx::TruncateToCapacity,
                                                topic_name)}) {
    this->allocator =
        (Allocator*) new ShareableAllocator((TopicHeader*) this->shm_base);
}

void Publisher::put(void* payload, size_t size) {
    if (!payload) throwError("Payload was not provided");
    if (size == 0) return;

    size_t offset = this->allocator->malloc(size);
    if (offset == -1) throwError("No free space in the pool");
    void* addr = (void*) ((uintptr_t) this->allocator->getPoolBase() + offset);

    this->allocator->copyTo(addr, payload, size);

    // get next available index as message id
    MessageQueueHeader* mq_header =
        getMessageQueueHeader(getTlsfHeader(getTopicHeader(this->shm_base)));
    size_t msg_id = std::atomic_ref<size_t>(mq_header->next).fetch_add(1) %
                    mq_header->capacity;
    MessageQueueEntry* mq_entry = getMessageQueueEntry(mq_header, msg_id);

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

    // notify subscribers with the message ID
    this->iox_publisher.publishCopyOf(msg_id).or_else([](auto& error) {
        std::stringstream ss;
        ss << "Iceoryx failed to send a message: " << error;
        throwError(ss.str().c_str());
    });
}