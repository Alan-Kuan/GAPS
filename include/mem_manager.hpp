#ifndef MEM_MANAGER_HPP
#define MEM_MANAGER_HPP

#include <cstddef>
#include <string>
#include <unordered_map>

#include <cuda.h>

class MemManager {
public:
    MemManager(const std::string& sock_file_dir = "/tmp/shoz");
    ~MemManager();

    [[noreturn]] void start();

private:
    struct Pool {
        CUmemGenericAllocationHandle handle;
        size_t size;
    };

    size_t createPool(const char* name, size_t size);
    void removePool(const char* name);

    size_t getPaddedSize(const size_t size,
                         const CUmemAllocationProp* prop) const;

    std::unordered_map<std::string, Pool> pools;
    std::string sock_file_dir;
    bool keep_running = true;
};

#endif  // MEM_MANAGER_HPP