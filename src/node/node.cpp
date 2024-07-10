#include "node/node.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cuda.h"

#include "alloc_algo/tlsf.hpp"
#include "error.hpp"
#include "metadata.hpp"

Node::Node(const char* topic_name, size_t pool_size, uint16_t domain_id) {
    if (strlen(topic_name) > kMaxTopicNameLen) throwError();

    // create new / attach existing shared memory
    size_t padded_pool_size = this->getPaddedSize(pool_size);
    size_t block_count = padded_pool_size / Tlsf::kBlockMinSize;

    size_t tlsf_size = sizeof(Tlsf::Header) + block_count * sizeof(Tlsf::BlockMetadata);
    size_t mq_size = sizeof(MessageQueueHeader) + kMaxMessageNum * sizeof(MessageQueueEntry);

    this->shm_size = sizeof(TopicHeader) + tlsf_size + sizeof(DomainMap) + mq_size;
    this->attachShm(topic_name, this->shm_size);

    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    Tlsf::Header* tlsf_header = getTlsfHeader(topic_header);
    DomainMap* domain_map = getDomainMap(tlsf_header);
    MessageQueueHeader* mq_header = getMessageQueueHeader(domain_map);

    // init the header if this is a newly created topic
    if (topic_header->topic_name[0] == '\0') {
        strcpy(topic_header->topic_name, topic_name);
        topic_header->pool_size = padded_pool_size;
    }
    std::atomic_ref<uint32_t>(topic_header->interest_count)++;

    // NOTE: if `pool_size` is not a multiple of `kBlockMinSize`, the remaining space will be wasted
    tlsf_header->aligned_pool_size = block_count * Tlsf::kBlockMinSize;
    tlsf_header->block_count = block_count;

    mq_header->capacity = kMaxMessageNum;

    // map unique ID of its domain to an index
    int i = 0;
    domain_map->lock.lock();
    for (; i < 32 && domain_map->map[i] > 0; i++) {
        if (domain_map->map[i] == domain_id) {
            domain_map->lock.unlock();
            this->domain_idx = i;
            return;
        }
    }
    // only supports 32 different domains currently 
    if (i == 32) {
        domain_map->lock.unlock();
        throwError("Too many different domains (supports only 32)");
    }
    domain_map->map[i] = domain_id;
    domain_map->lock.unlock();
    this->domain_idx = i;
}

Node::~Node() {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    // TODO: make it a critical section, i.e., make sure no other node just joins
    // right after the condition is met but before unlinking the file
    if (std::atomic_ref<uint32_t>(topic_header->interest_count).fetch_sub(1) == 1) {
        throwOnError(shm_unlink(topic_header->topic_name));
    }

    // the allocator is constructed in the derived class
    delete this->allocator;
    this->detachShm(this->shm_size);
}

void Node::attachShm(const char* shm_name, size_t size) {
    int fd = throwOnError(shm_open(shm_name, O_CREAT | O_RDWR, 0666));
    throwOnError(ftruncate(fd, size));

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) throwError();
    this->shm_base = ptr;

    throwOnError(close(fd));
}

void Node::detachShm(size_t size) {
    throwOnError(munmap(this->shm_base, size));
}

// NOTE: sizes of the pools of all types of memory domains should be the same,
//       so this function should be called every time
size_t Node::getPaddedSize(const size_t size) const {
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    size_t gran = 0;
    throwOnErrorCuda(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return ((size - 1) / gran + 1) * gran;
}