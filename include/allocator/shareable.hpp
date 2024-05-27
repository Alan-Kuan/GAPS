#ifndef SHAREABLE_ALLOCATOR_HPP
#define SHAREABLE_ALLOCATOR_HPP

#include <cstddef>

#include <cuda.h>

#include "alloc_algo/tlsf.hpp"
#include "allocator/allocator.hpp"

class ShareableAllocator : public Allocator {
public:
    ShareableAllocator() = delete;
    ShareableAllocator(void* metadata, size_t pool_size);
    ShareableAllocator(void* metadata);
    ~ShareableAllocator();

    void* malloc(size_t size) override;
    void free(void* addr) override;
    void copyTo(void* dst, void* src, size_t size) override;
    void copyFrom(void* dst, void* src, size_t size) override;

    void shareHandle(int count);
    void recvHandle();

private:
    typedef int ShareableHandle;

    void createPool(size_t size) override;
    void attachPool(bool read_only);
    void detachPool();

    size_t getPaddedSize(const size_t size, const CUmemAllocationProp* prop) const;

    CUmemGenericAllocationHandle handle;
    bool handle_is_valid = false;
    Metadata* metadata = nullptr;
    Tlsf* allocator = nullptr;
};

#endif  // SHAREABLE_ALLOCATOR_HPP