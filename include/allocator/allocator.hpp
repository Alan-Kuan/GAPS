#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <cstddef>
#include <string>

#include <cuda.h>

#include "metadata.hpp"

class Allocator {
public:
    friend class Node;

    Allocator() = delete;
    Allocator(TopicHeader* topic_header, bool read_only = false,
              const std::string& sock_file_dir = "/tmp/shoi");
    virtual ~Allocator();

    virtual size_t malloc(size_t size) = 0;
    virtual void free(size_t offset) = 0;

    void* getPoolBase() const;

private:
    void createPool(size_t size);
    void detachPool();
    void removePool();
    int connectServer();
    void disconnectServer(int sockfd);
    void recvHandle();

    void* pool_base = nullptr;
    TopicHeader* topic_header = nullptr;
    bool read_only = false;

    std::string sock_file_dir;
    CUmemGenericAllocationHandle handle;
    bool handle_is_valid = false;
};

#endif  // ALLOCATOR_HPP