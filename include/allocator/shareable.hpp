#ifndef SHAREABLE_ALLOCATOR_HPP
#define SHAREABLE_ALLOCATOR_HPP

#include <cstddef>
#include <string>

#include <cuda.h>

#include "allocator/allocator.hpp"
#include "metadata.hpp"

class ShareableAllocator : public Allocator {
public:
    ShareableAllocator() = delete;
    ShareableAllocator(TopicHeader* topic_header, bool read_only = false,
        const std::string& sock_file_dir = "/tmp/shoz");
    ~ShareableAllocator();

    void copyTo(void* dst, void* src, size_t size) override;
    void copyFrom(void* dst, void* src, size_t size) override;

private:
    void createPool(size_t size) override;
    void removePool();
    void recvHandle();

    std::string sock_file_dir;
    CUmemGenericAllocationHandle handle;
    bool handle_is_valid = false;
};

#endif  // SHAREABLE_ALLOCATOR_HPP