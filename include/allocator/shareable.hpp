#ifndef SHAREABLE_ALLOCATOR_HPP
#define SHAREABLE_ALLOCATOR_HPP

#include <cstddef>

#include <cuda.h>

#include "allocator/base.hpp"

class ShareableAllocator : public Allocator {
private:
    typedef int ShareableHandle;
    struct Metadata {
        char topic_name[32];
        size_t pool_size;
    };

    void createPool(size_t size);
    void attachPool(void);
    void detachPool(void);

    size_t getPaddedSize(const size_t size, const CUmemAllocationProp* prop) const;
    inline Metadata* getMetadata(void) const;

public:
    ShareableAllocator(const char* topic_name, size_t pool_size);
    ShareableAllocator(const char* topic_name);
    ~ShareableAllocator(void);

    void shareHandle(int count);
    void recvHandle(void);

    void* malloc(size_t size);
    void free(void* ptr);

private:
    CUmemGenericAllocationHandle handle;
};

#endif  // SHAREABLE_ALLOCATOR_HPP