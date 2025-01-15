#ifndef METADATA_HPP
#define METADATA_HPP

#include <semaphore.h>

#include <cstddef>
#include <cstdint>

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
 *  - TLSF Section:
 *  ┌────────┬─────────────────────────┐
 *  │ Header │ Array of Block Metadata │
 *  └────────┴─────────────────────────┘
 *
 *  - Message Queue Section:
 *  ┌────────┬────────────────────────────────┐
 *  │ Header │ Array of Message Queue Entries │
 *  └────────┴────────────────────────────────┘
 */

/**
 *  Topic Header
 */

struct TopicHeader {
    char topic_name[32];
    size_t pool_size;
    // number of publishers and subscribers (should be atomic referenced)
    uint32_t interest_count;
    // number of subscribers (should be atomic referenced)
    uint32_t sub_count;
};

/**
 *  TLSF Section
 */

const int kFstLvlCnt = 32;
const int kSndLvlIdx = 4;
constexpr int kSndLvlCnt = 1 << kSndLvlIdx;
// since the minimum block size is 2^4, the header can contain at most 4 special
// flags
const size_t kBlockMinSize = 16;

const size_t kBlockFlagBits = 0b11;
const size_t kBlockFreeFlag = 0b01;
const size_t kBlockLastFlag = 0b10;

struct TlsfHeader {
    sem_t lock;
    // whether the pool has been initialized (should be atomic referenced)
    bool inited;
    // aligned to a multiple of the minimum block size
    size_t aligned_pool_size;
    // number of blocks in this pool
    size_t block_count;
    // indicate if any block exists in the group of lists indexed by `fidx`
    uint32_t first_lvl;
    // indicate if any block exists in the list indexed by `fidx` and `sidx`
    uint32_t second_lvl[kFstLvlCnt];
    // free lists of blocks (saved in the form of "block_idx + 1")
    size_t free_lists[kFstLvlCnt][kSndLvlCnt];
};

struct TlsfBlockMetadata {
    size_t getSize() const;
    // block size and flags (bit 0 is free flag, bit 1 is last flag)
    size_t header;
    size_t phys_prev_idx;
    size_t prev_free_idx;
    size_t next_free_idx;
};

/**
 *  Message Queue Section
 */

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

/**
 *  Inline Utility Functions
 */

inline uint16_t Domain::getId() const {
    // +1 to preserve 0 as undefined
    return dev_id * 10 + (uint16_t) dev_type + 1;
}

inline TopicHeader* getTopicHeader(void* shm_base) {
    return (TopicHeader*) shm_base;
}

inline TlsfHeader* getTlsfHeader(TopicHeader* topic_header) {
    return (TlsfHeader*) ((uintptr_t) topic_header + sizeof(TopicHeader));
}

inline size_t TlsfBlockMetadata::getSize() const {
    return this->header & ~kBlockFlagBits;
}

// WARN: this function depends on tlsf_header->block_count, so it should be set
// before this function is used
inline MessageQueueHeader* getMessageQueueHeader(TlsfHeader* tlsf_header) {
    return (MessageQueueHeader*) ((uintptr_t) tlsf_header + sizeof(TlsfHeader) +
                                  tlsf_header->block_count *
                                      sizeof(TlsfBlockMetadata));
}

inline MessageQueueEntry* getMessageQueueEntry(MessageQueueHeader* mq_header,
                                               size_t idx) {
    return (MessageQueueEntry*) ((uintptr_t) mq_header +
                                 sizeof(MessageQueueHeader) +
                                 idx * sizeof(MessageQueueEntry));
}

#ifdef BUILD_PYSHOZ
#include <nanobind/ndarray.h>

/**
 *  Message:
 *  ┌────────┬─────────────────────────┐
 *  │ Header │ Shape (size_t[ndim])    │
 *  └────────┴─────────────────────────┘
 */

struct MsgHeader {
    size_t msg_id;
    nanobind::dlpack::dtype dtype;
    int32_t ndim;
};
#endif

#endif  // METADATA_HPP