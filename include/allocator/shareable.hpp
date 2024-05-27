#ifndef SHAREABLE_ALLOCATOR_HPP
#define SHAREABLE_ALLOCATOR_HPP

#include <cstddef>

#include <cuda.h>

#include "alloc_algo/tlsf.hpp"
#include "allocator/allocator.hpp"

class ShareableAllocator : public Allocator {
public:
    ShareableAllocator(void) = delete;
    ShareableAllocator(void* metadata, size_t pool_size);
    ShareableAllocator(void* metadata);
    ~ShareableAllocator(void);

    void* malloc(size_t size);
    void free(void* addr);
    void copyTo(void* dst, void* src, size_t size);
    void copyFrom(void* dst, void* src, size_t size);

    void shareHandle(int count);
    void recvHandle(void);

private:
    typedef int ShareableHandle;

    void createPool(size_t size);
    void attachPool(bool read_only);
    void detachPool(void);

    size_t getPaddedSize(const size_t size, const CUmemAllocationProp* prop) const;

    CUmemGenericAllocationHandle handle;
    bool handle_is_valid = false;
    Metadata* metadata = nullptr;
    Tlsf* allocator = nullptr;
};

#endif  // SHAREABLE_ALLOCATOR_HPP