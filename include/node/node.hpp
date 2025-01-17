#ifndef NODE_HPP
#define NODE_HPP

#include <cstddef>

#include "allocator/allocator.hpp"

class Node {
public:
    Node() = delete;
    Node(const char* topic_name, size_t pool_size, int msg_queue_cap_exp);
    ~Node();
    // prevent the node from being copied, since it may cause problem when the
    // copy destructs
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

protected:
    static const int kMaxTopicNameLen = 31;

    void* shm_base = nullptr;
    size_t shm_size = 0;
    Allocator* allocator = nullptr;

private:
    void attachShm(const char* shm_name, size_t size);
    void detachShm(size_t size);
    size_t getPaddedSize(const size_t size) const;
};

#endif  // NODE_HPP