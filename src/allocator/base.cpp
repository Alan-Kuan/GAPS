#include "allocator/base.hpp"

#include "errno.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <string>
#include <source_location>

#include "error.hpp"

void Allocator::attachShm(const char* shm_name, size_t size) {
    int fd = throwOnError(shm_open(shm_name, O_CREAT | O_RDWR, 0666));
    throwOnError(ftruncate(fd, size));

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) throwError();
    this->shm_base = ptr;
}

// NOTE: the reason why `std::string` is used is to deep copy
//       the name before the shared memory gets unmapped
void Allocator::detachShm(std::string shm_name, size_t size) {
    throwOnError(munmap(this->shm_base, size));
    // TODO: maintain a ref count in shared memory; once it reduced to 0, call unlink
    // unlinking an shared memory file that has already been unlinked is acceptable
    if (shm_unlink(shm_name.c_str()) != 0 && errno != ENOENT) throwError();
}