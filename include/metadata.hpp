#ifndef METADATA_HPP
#define METADATA_HPP

#include <cstddef>
#include <cstdint>

#include "alloc_algo/tlsf.hpp"

enum class DeviceType : uint8_t { kHost, kGPU };

struct Domain {
    uint16_t getId() const;
    DeviceType dev_type;
    uint16_t dev_id;
};

/**
 *  Shared Metadata Store:
 *  ┌──────────────┬──────────────┬───────────────────────┐
 *  │ Topic Header │ TLSF Section │ Message Queue Section │
 *  └──────────────┴──────────────┴───────────────────────┘
 *
 *  TLSF Section:
 *  ┌────────┬─────────────────────────┐
 *  │ Header │ Array of Block Metadata │
 *  └────────┴─────────────────────────┘
 *
 *  Message Queue Section:
 *  ┌────────┬────────────────────────────────┐
 *  │ Header │ Array of Message Queue Entries │
 *  └────────┴────────────────────────────────┘
 */

struct TopicHeader {
    char topic_name[32];
    size_t pool_size;
    // number of publishers and subscribers (should be atomic referenced)
    uint32_t interest_count;
    // number of subscribers (should be atomic referenced)
    uint32_t sub_count;
};

struct MessageQueueHeader {
    size_t capacity;
    // indicates the next index available to put the message (should be atomic
    // referenced)
    size_t next;
};

struct MessageQueueEntry {
    // number of subscribers that have taken the payload (should be atomic
    // referenced)
    uint32_t taken_num;
    // address offset of the allocated space for the payload
    // (its first bit is used to determine if the relevant space is freed)
    size_t offset;
    // size of the payload
    size_t size;
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

inline MessageQueueHeader* getMessageQueueHeader(Tlsf::Header* tlsf_header) {
    return (MessageQueueHeader*) ((uintptr_t) tlsf_header +
                                  sizeof(Tlsf::Header) +
                                  tlsf_header->block_count *
                                      sizeof(Tlsf::BlockMetadata));
}

inline MessageQueueEntry* getMessageQueueEntry(MessageQueueHeader* mq_header,
                                               size_t idx) {
    return (MessageQueueEntry*) ((uintptr_t) mq_header +
                                 sizeof(MessageQueueHeader) +
                                 idx * sizeof(MessageQueueEntry));
}

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>

struct MsgBuf {
    size_t msg_id;
    nanobind::dlpack::dtype dtype;
    int32_t ndim;
    int64_t shape[3];
    int64_t strides[3];
};
#endif

#endif  // METADATA_HPP