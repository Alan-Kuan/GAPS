#ifndef TLSF_HPP
#define TLSF_HPP

#include <semaphore.h>

#include <cstddef>

#include "allocator/allocator.hpp"
#include "metadata.hpp"

class TlsfAllocator : public Allocator {
public:
    TlsfAllocator() = delete;
    TlsfAllocator(TopicHeader* topic_header, bool read_only = false,
                  const std::string& sock_file_dir = "/tmp/shoz");

    size_t malloc(size_t size);
    void free(size_t offset);

private:
    // bit number of size_t - 1
    static constexpr int kWidthMinusOne = sizeof(size_t) * 8 - 1;

    void mapping(size_t size, int* fidx, int* sidx) const;
    size_t alignSize(size_t size) const;

    size_t findSuitableBlock(size_t size, int* fidx, int* sidx);

    size_t splitBlock(size_t idx, size_t size);
    size_t mergeBlock(size_t idx);

    void insertBlock(size_t idx);
    void removeBlock(size_t idx);
    void removeBlock(size_t idx, int fidx, int sidx);

    TlsfHeader* tlsf_header = nullptr;
    TlsfBlockMetadata* blocks = nullptr;
};

#endif  // TLSF_HPP