#include "node/node.hpp"

#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda.h>

#include "error.hpp"
#include "metadata.hpp"

Node::Node(const char* topic_name, size_t pool_size, int msg_queue_cap_exp) {
    if (strlen(topic_name) > kMaxTopicNameLen) {
        throwError("topic name is too long");
    }

    // init CUDA Driver API
    throwOnErrorCuda(cuInit(0));

    /**
     *  Create a new / attach an existing shared metadata store
     */

    size_t padded_pool_size = this->getPaddedSize(pool_size);
    size_t block_count = padded_pool_size / kBlockMinSize;
    size_t msg_queue_cap = 1 << msg_queue_cap_exp;

    size_t tlsf_size =
        sizeof(TlsfHeader) + block_count * sizeof(TlsfBlockMetadata);
    size_t mq_size =
        sizeof(MessageQueueHeader) + msg_queue_cap * sizeof(MessageQueueEntry);

    this->shm_size = sizeof(TopicHeader) + tlsf_size + mq_size;
    this->attachShm(topic_name, this->shm_size);

    /**
     *  Init headers
     */

    TopicHeader* topic_header = getTopicHeader(this->shm_base);

    // WARN: there may be a race condition if 2 nodes of the same topic are
    // created almost at the same time

    // init the header if this is a newly created topic
    if (topic_header->topic_name[0] == '\0') {
        strcpy(topic_header->topic_name, topic_name);
        topic_header->pool_size = padded_pool_size;

        TlsfHeader* tlsf_header = getTlsfHeader(topic_header);
        // NOTE: if `pool_size` is not a multiple of `kBlockMinSize`, the
        // remaining space will be wasted
        tlsf_header->aligned_pool_size = block_count * kBlockMinSize;
        tlsf_header->block_count = block_count;
        throwOnError(sem_init(&tlsf_header->lock, 1, 1));

        MessageQueueHeader* mq_header = getMessageQueueHeader(tlsf_header);
        mq_header->capacity = msg_queue_cap;
    } else {
        if (topic_header->pool_size != padded_pool_size) {
            throwError("pool size does not match");
        }
        MessageQueueHeader* mq_header =
            getMessageQueueHeader(getTlsfHeader(topic_header));
        if (mq_header->capacity != msg_queue_cap) {
            throwError("message queue capacity does not match");
        }
    }
    std::atomic_ref<uint32_t>(topic_header->interest_count)++;
}

Node::~Node() {
    TopicHeader* topic_header = getTopicHeader(this->shm_base);
    // TODO: make it a critical section, i.e., make sure no other node just
    // joins right after the condition is met but before unlinking the file
    if (std::atomic_ref<uint32_t>(topic_header->interest_count).fetch_sub(1) ==
        1) {
        throwOnError(shm_unlink(topic_header->topic_name));
        this->allocator->removePool();
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

// NOTE: sizes of the pools of all kinds of memory domains should be the same,
//       so this function should be called every time
size_t Node::getPaddedSize(const size_t size) const {
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    size_t gran = 0;
    throwOnErrorCuda(cuMemGetAllocationGranularity(
        &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return ((size - 1) / gran + 1) * gran;
}