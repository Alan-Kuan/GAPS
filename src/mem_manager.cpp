#include "mem_manager.hpp"

#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <cuda.h>

#include "error.hpp"

MemManager* manager = nullptr;

int main(int argc, char* argv[]) {
    manager = new MemManager(argc > 1 ? argv[1] : "/tmp/shoz");

    signal(SIGINT, [](int _) {
        delete manager;
        std::cout << "Stopped" << std::endl;
        exit(0);
    });

    std::cout << "Starting shareable GPU memory manager" << std::endl;
    try {
        if (cuInit(0) != CUDA_SUCCESS) throwError();
        manager->start();
    } catch (std::runtime_error& err) {
        std::cerr << "Memory Manager: " << err.what() << std::endl;
        return 1;
    }

    return 0;
}

MemManager::MemManager(const std::string& sock_file_dir) : sock_file_dir(sock_file_dir) {
    std::filesystem::create_directory(sock_file_dir);
}

MemManager::~MemManager() {
    char sock_file_path[108];
    sprintf(sock_file_path, "%s/server.sock", this->sock_file_dir.c_str());
    throwOnError(unlink(sock_file_path));

    for (auto& entry : this->pools) {
        throwOnErrorCuda(cuMemRelease(entry.second.handle));
    }
}

[[noreturn]] void MemManager::start() {
    // setup a UNIX Domain Socket server
    int sockfd = throwOnError(socket(AF_UNIX, SOCK_STREAM, 0));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    sprintf(addr.sun_path, "%s/server.sock", this->sock_file_dir.c_str());

    throwOnError(bind(sockfd, (struct sockaddr*) &addr, sizeof(addr)));
    throwOnError(listen(sockfd, 8));

    // prepare buffers for messages
    struct {
        size_t pool_size;
        char topic_name[32];
    } buf_req;
    struct msghdr msg;
    struct iovec iov[1];
    char ctrl_buf[CMSG_SPACE(sizeof(int))];

    msg.msg_name = nullptr;
    msg.msg_namelen = 0;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctrl_buf;
    msg.msg_controllen = sizeof(ctrl_buf);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);

    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;

    while (true) {
        size_t padded_size;
        int cli_fd = throwOnError(accept(sockfd, nullptr, nullptr));

        // receive request
        throwOnError(recv(cli_fd, &buf_req, sizeof(buf_req), 0));

        auto entry = this->pools.find(buf_req.topic_name);
        if (entry == this->pools.end()) {
            padded_size = this->createPool(buf_req.topic_name, buf_req.pool_size);
        } else {
            padded_size = entry->second.size;
        }

        // export the handle to a shareable handle
        int sh_handle;
        throwOnErrorCuda(cuMemExportToShareableHandle((void*) &sh_handle,
            this->pools[buf_req.topic_name].handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

        iov[0].iov_base = &padded_size;
        iov[0].iov_len = sizeof(size_t);
        *((int*) CMSG_DATA(cmsg)) = sh_handle;

        throwOnError(sendmsg(cli_fd, &msg, 0));
        throwOnError(close(cli_fd));
    }
}

size_t MemManager::createPool(const char* name, size_t size) {
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    Pool pool;
    size_t padded_size = this->getPaddedSize(size, &prop);
    throwOnErrorCuda(cuMemCreate(&pool.handle, padded_size, &prop, 0));
    pool.size = padded_size;
    this->pools[name] = pool;

    return padded_size;
}

void MemManager::removePool(const char* name) {
    throwOnErrorCuda(cuMemRelease(this->pools[name].handle));
    this->pools.erase(name);
}

size_t MemManager::getPaddedSize(const size_t size, const CUmemAllocationProp* prop) const {
    size_t gran = 0;
    throwOnErrorCuda(cuMemGetAllocationGranularity(&gran, prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return ((size - 1) / gran + 1) * gran;
}