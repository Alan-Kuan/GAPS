#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <cstddef>
#include <string>

#include <cuda.h>

#include "alloc_algo/tlsf.hpp"
#include "metadata.hpp"

class Allocator {
public:
    Allocator() = delete;
    Allocator(TopicHeader* topic_header, bool read_only = false,
              const std::string& sock_file_dir = "/tmp/shoz");
    ~Allocator();

    size_t malloc(size_t size);
    void free(size_t offset);

    void* getPoolBase() const { return this->pool_base; }

private:
    void createPool(size_t size);
    void removePool();
    int connectServer();
    void disconnectServer(int sockfd);
    void recvHandle();

    void* pool_base = nullptr;
    TopicHeader* topic_header = nullptr;
    bool read_only = false;
    Tlsf* allocator = nullptr;

    std::string sock_file_dir;
    CUmemGenericAllocationHandle handle;
    bool handle_is_valid = false;
};

#endif  // ALLOCATOR_HPP