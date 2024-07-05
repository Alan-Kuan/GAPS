#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstddef>
#include <cstdint>

#include "alloc_algo/tlsf.hpp"
#include "ticket_lock.hpp"

enum DeviceType { kHost, kGPU };

struct Domain {
    uint16_t getId() const;
    DeviceType dev_type;
    uint16_t dev_id;
};

struct TopicHeader {
    char topic_name[32];
    size_t pool_size;
    // number of publishers and subscribers (should be atomic referenced)
    uint32_t interest_count;
    // number of subscribers (should be atomic referenced)
    uint32_t sub_count;
};

struct DomainMap {
    // this is a spin lock; keep the critical section tiny
    TicketLock lock;
    // index: domain_idx, value: domain_id
    uint16_t map[32];
};

struct MessageQueueHeader {
    size_t capacity;
    // indicates the next index available to put the message (should be atomic referenced)
    size_t next;
};

struct MessageQueueEntry {
    // number of subscribers that have taken the payload (should be atomic referenced)
    uint32_t taken_num;
    // address offset of the allocated space for the payload
    size_t offset;
    // size of the payload
    size_t size;
    // a bitmap indicates whether the payload exists in the corresponding domain (should be atomic referenced)
    uint32_t avail;
};

inline uint16_t Domain::getId() const {
    // +1 to preserve 0 as undefined
    return dev_id * 10 + (uint16_t) dev_type + 1;
}

inline TopicHeader* getTopicHeader(void* shm_base) {
    return (TopicHeader*) shm_base;
}

inline Tlsf::Header* getTlsfHeader(TopicHeader* topic_header) {
    return (Tlsf::Header*) ((uintptr_t) topic_header + sizeof(TopicHeader));
}

inline DomainMap* getDomainMap(Tlsf::Header* tlsf_header) {
    return (DomainMap*) ((uintptr_t) tlsf_header + sizeof(Tlsf::Header) + tlsf_header->block_count * sizeof(Tlsf::BlockMetadata));
}

inline MessageQueueHeader* getMessageQueueHeader(DomainMap* domain_map) {
    return (MessageQueueHeader*) ((uintptr_t) domain_map + sizeof(DomainMap));
}

inline MessageQueueEntry* getMessageQueueEntry(MessageQueueHeader* mq_header, size_t idx) {
    return (MessageQueueEntry*) ((uintptr_t) mq_header + sizeof(MessageQueueHeader) + idx * sizeof(MessageQueueEntry));
}

#endif  // TYPES_HPP