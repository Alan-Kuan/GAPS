#ifndef SHAREABLE_ALLOCATOR_HPP
#define SHAREABLE_ALLOCATOR_HPP

#include <cstddef>
#include <string>

#include <cuda.h>

#include "alloc_algo/tlsf.hpp"
#include "allocator/allocator.hpp"

class ShareableAllocator : public Allocator {
public:
    ShareableAllocator() = delete;
    ShareableAllocator(void* metadata, size_t pool_size, bool read_only = false,
        const std::string& sock_file_dir = "/tmp/shoz");
    ~ShareableAllocator();

    void* malloc(size_t size) override;
    void free(void* addr) override;
    void copyTo(void* dst, void* src, size_t size) override;
    void copyFrom(void* dst, void* src, size_t size) override;

private:
    void createPool(size_t size) override;
    void removePool();
    size_t recvHandle(size_t pool_size);

    std::string sock_file_dir;
    CUmemGenericAllocationHandle handle;
    bool handle_is_valid = false;
    Metadata* metadata = nullptr;
    Tlsf* allocator = nullptr;
};

#endif  // SHAREABLE_ALLOCATOR_HPP