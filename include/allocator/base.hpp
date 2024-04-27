#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <cstddef>
#include <string>

class Allocator {
protected:
    virtual void createPool(size_t size) = 0;

public:
    void attachShm(const char* shm_name, size_t size);
    void detachShm(std::string shm_name, size_t size);

    virtual void* malloc(size_t size) = 0;
    virtual void free(void* ptr) = 0;

protected:
    void* pool_base = nullptr;
    void* shm_base = nullptr;
};

#endif  // ALLOCATOR_HPP