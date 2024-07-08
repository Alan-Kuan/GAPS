#include "node/node.hpp"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstring>
#include <string>

#include "allocator/allocator.hpp"
#include "error.hpp"

Node::Node(const char* topic_name) {
    if (strlen(topic_name) > kMaxTopicNameLen) throwError();

    this->shm_size =
        sizeof(Allocator::Metadata) +
        // TODO: we assume subscribers from all memory domains here; can be on
        // demand in the future
        sizeof(MessageQueueHeader) +
        kMaxMessageNum * (sizeof(int) + kMaxDomainNum * sizeof(size_t));
    this->attachShm(topic_name, this->shm_size);

    if (((char*) this->shm_base)[0] == '\0') {
        strcpy((char*) this->shm_base, topic_name);
    }
}

Node::~Node() { this->detachShm((char*) this->shm_base, this->shm_size); }

void Node::attachShm(const char* shm_name, size_t size) {
    int fd = throwOnError(shm_open(shm_name, O_CREAT | O_RDWR, 0666));
    throwOnError(ftruncate(fd, size));

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) throwError();
    this->shm_base = ptr;

    throwOnError(close(fd));
}

// NOTE: the reason why `std::string` is used is to deep copy
//       the name before the shared memory gets unmapped
void Node::detachShm(std::string shm_name, size_t size) {
    throwOnError(munmap(this->shm_base, size));
    // TODO: let the last publisher who unmaps the memory to unlink
    // it's ok if the file has already been deleted by some node
    if (shm_unlink(shm_name.c_str()) != 0 && errno != ENOENT) throwError();
}