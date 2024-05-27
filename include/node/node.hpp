#ifndef NODE_HPP
#define NODE_HPP

#include <cstddef>
#include <string>

#include "allocator/allocator.hpp"

class Node {
public:
    struct MessageQueueHeader {
        size_t capacity;
        // indicates the next index available to put the message (should be atomic referenced)
        size_t next;
        // number of subscribers (should be atomic referenced)
        uint32_t sub_count;
    };

    Node() = delete;
    Node(const char* topic_name);
    ~Node();
    // prevent the node from being copied, since it may cause problem when the copy destructs
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

protected:
    static const int kMaxTopicNameLen = 31;
    static const int kMaxDomainNum = 2;
    static const size_t kMaxMessageNum = 128;  // NOTE: it must be power of 2

    void* shm_base = nullptr;
    size_t shm_size = 0;
    Allocator* allocator = nullptr;

private:
    void attachShm(const char* shm_name, size_t size);
    void detachShm(std::string shm_name, size_t size);
};

#endif  // NODE_HPP