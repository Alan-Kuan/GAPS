#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <cstddef>

#include "alloc_algo/tlsf.hpp"
#include "metadata.hpp"

class Allocator {
public:
    Allocator(TopicHeader* topic_header, bool read_only) : topic_header(topic_header), read_only(read_only) {}
    virtual ~Allocator() {}

    virtual size_t malloc(size_t size);
    virtual void free(size_t offset);
    // copy to pool (dst) from host (src)
    virtual void copyTo(void* dst, void* src, size_t size) = 0;
    // copy from pool (src) to host (dst)
    virtual void copyFrom(void* dst, void* src, size_t size) = 0;

    void* getPoolBase() const { return this->pool_base; }

protected:
    virtual void createPool(size_t size) = 0;

    void* pool_base = nullptr;
    TopicHeader* topic_header = nullptr;
    bool read_only = false;
    Tlsf* allocator = nullptr;
};

#endif  // ALLOCATOR_HPP