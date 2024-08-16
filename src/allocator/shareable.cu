#include "allocator/shareable.hpp"

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include <cuda.h>

#include "alloc_algo/tlsf.hpp"
#include "allocator/allocator.hpp"
#include "error.hpp"
#include "metadata.hpp"

ShareableAllocator::ShareableAllocator(TopicHeader* topic_header,
                                       bool read_only,
                                       const std::string& sock_file_dir)
        : Allocator(topic_header, read_only), sock_file_dir(sock_file_dir) {
    std::filesystem::create_directory(sock_file_dir);
    this->createPool(topic_header->pool_size);
    this->allocator = new Tlsf(getTlsfHeader(topic_header));
}

ShareableAllocator::~ShareableAllocator() { this->removePool(); }

void ShareableAllocator::copyTo(void* dst, void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void ShareableAllocator::copyFrom(void* dst, void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void ShareableAllocator::createPool(size_t size) {
    this->recvHandle();

    CUdeviceptr dptr;
    throwOnErrorCuda(cuMemAddressReserve(&dptr, size, 0, 0, 0));
    throwOnErrorCuda(cuMemMap(dptr, size, 0, this->handle, 0));

    CUmemAccessDesc acc_desc;
    acc_desc.location.id = 0;
    acc_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc_desc.flags = this->read_only ? CU_MEM_ACCESS_FLAGS_PROT_READ
                                     : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    throwOnErrorCuda(cuMemSetAccess(dptr, size, &acc_desc, 1));

    this->pool_base = (void*) dptr;
}

void ShareableAllocator::removePool() {
    if (!this->handle_is_valid) return;
    throwOnErrorCuda(cuMemRelease(this->handle));
    throwOnErrorCuda(cuMemUnmap((CUdeviceptr) this->pool_base,
                                this->topic_header->pool_size));
    throwOnErrorCuda(cuMemAddressFree((CUdeviceptr) this->pool_base,
                                      this->topic_header->pool_size));
}

void ShareableAllocator::recvHandle() {
    // setup UNIX Domain Socket client
    int sockfd = throwOnError(socket(AF_UNIX, SOCK_STREAM, 0));

    struct sockaddr_un cli_addr;
    memset(&cli_addr, 0, sizeof(cli_addr));
    cli_addr.sun_family = AF_UNIX;
    throwOnError(sprintf(cli_addr.sun_path, "%s/%s-client-%d-%d.sock",
                         this->sock_file_dir.c_str(),
                         this->topic_header->topic_name, getpid(), gettid()));

    throwOnError(bind(sockfd, (struct sockaddr*) &cli_addr, sizeof(cli_addr)));

    struct sockaddr_un server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;
    throwOnError(sprintf(server_addr.sun_path, "%s/server.sock",
                         this->sock_file_dir.c_str()));

    // should wait until the server is ready
    struct stat buf;
    for (int tries = 0; tries < 2 && (stat(server_addr.sun_path, &buf) != 0);
         tries++) {
        usleep(50000);
    }
    throwOnError(
        connect(sockfd, (struct sockaddr*) &server_addr, sizeof(server_addr)));

    // send a request for the handle of a memory pool
    struct {
        size_t pool_size;
        char topic_name[32];
    } buf_req;

    buf_req.pool_size = this->topic_header->pool_size;
    strcpy(buf_req.topic_name, this->topic_header->topic_name);

    throwOnError(send(sockfd, &buf_req, sizeof(buf_req), 0));

    // receive the message with the shareable handle
    struct msghdr msg;
    struct iovec iov[1];
    char ctrl_buf[CMSG_SPACE(sizeof(int))];
    char dummy;

    iov[0].iov_base = &dummy;
    iov[0].iov_len = 1;

    msg.msg_name = nullptr;
    msg.msg_namelen = 0;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctrl_buf;
    msg.msg_controllen = sizeof(ctrl_buf);

    throwOnError(recvmsg(sockfd, &msg, 0));

    throwOnError(close(sockfd));
    throwOnError(unlink(cli_addr.sun_path));

    // import the shareable handle into a generic handle
    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_len != CMSG_LEN(sizeof(int))) throwError();

    int sh_handle = *((int*) CMSG_DATA(cmsg));
    throwOnErrorCuda(cuMemImportFromShareableHandle(
        &this->handle, (void*) (uintptr_t) sh_handle,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    this->handle_is_valid = true;
}