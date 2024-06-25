#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <cstddef>

class Allocator {
public:
    enum DeviceType { kHost, kGPU };
    struct Domain {
        DeviceType dev_type;
        int dev_id;
        // TODO: also need a way to convert it to index
        int getId() const { return dev_id * 10 + (int) dev_type; }
    };

    struct Metadata {
        char topic_name[32];
        size_t pool_size;
    };

    Allocator(bool read_only) : read_only(read_only) {}
    virtual ~Allocator() {}

    virtual void* malloc(size_t size) = 0;
    virtual void free(void* ptr) = 0;
    // copy to pool (dst) from host (src)
    virtual void copyTo(void* dst, void* src, size_t size) = 0;
    // copy from pool (src) to host (dst)
    virtual void copyFrom(void* dst, void* src, size_t size) = 0;

    void* getPoolBase() const { return this->pool_base; }

protected:
    virtual void createPool(size_t size) = 0;

    void* pool_base = nullptr;
    bool read_only = false;
};

#endif  // ALLOCATOR_HPP