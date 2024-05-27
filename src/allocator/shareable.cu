#include "allocator/shareable.hpp"

#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <source_location>

#include <cuda.h>

#include "alloc_algo/tlsf.hpp"
#include "error.hpp"

namespace {
void throwOnErrorCuda(CUresult res, std::source_location loc = std::source_location::current()) {
    if (res != CUDA_SUCCESS) {
        const char* msg;
        cuGetErrorString(res, &msg);
        throwError(msg, loc);
    }
}
}

ShareableAllocator::ShareableAllocator(void* metadata, size_t pool_size) {
    this->metadata = (Metadata*) metadata;
    this->createPool(pool_size);
    this->attachPool(false);
    this->allocator = new Tlsf(this->pool_base, pool_size);
}

ShareableAllocator::ShareableAllocator(void* metadata) {
    this->metadata = (Metadata*) metadata;
}

ShareableAllocator::~ShareableAllocator() {
    this->detachPool();
}

void* ShareableAllocator::malloc(size_t size) {
    if (!this->allocator) throwError("No allocator");
    return this->allocator->malloc(size);
}

void ShareableAllocator::free(void* addr) {
    if (!this->allocator) throwError("No allocator");
    this->allocator->free(addr);
}

void ShareableAllocator::copyTo(void* dst, void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void ShareableAllocator::copyFrom(void* dst, void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void ShareableAllocator::shareHandle(int count) {
    // export the handle to a shareable handle
    ShareableHandle sh_handle;
    throwOnErrorCuda(cuMemExportToShareableHandle((void*) &sh_handle,
        this->handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

    // setup a UNIX Domain Socket server
    int sockfd = throwOnError(socket(AF_UNIX, SOCK_STREAM, 0));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    throwOnError(sprintf(addr.sun_path, "/tmp/shoz/%s-server.sock",
        this->metadata->topic_name));

    throwOnError(bind(sockfd, (struct sockaddr*) &addr, sizeof(addr)));
    throwOnError(listen(sockfd, 8));

    // prepare message
    struct msghdr msg;
    struct iovec iov[1];
    char ctrl_buf[CMSG_SPACE(sizeof(int))];

    // dummy data
    iov[0].iov_base = (void*) "";
    iov[0].iov_len = 1;

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
    *((int*) CMSG_DATA(cmsg)) = (int) sh_handle;

    // send shareable handle to `count` clients
    for (int N = count; N > 0; N--) {
        int cli_fd = throwOnError(accept(sockfd, nullptr, nullptr));
        throwOnError(sendmsg(cli_fd, &msg, 0));
        throwOnError(close(cli_fd));
    }

    throwOnError(close(sockfd));
    throwOnError(unlink(addr.sun_path));
}

void ShareableAllocator::recvHandle() {
    // setup UNIX Domain Socket client
    int sockfd = throwOnError(socket(AF_UNIX, SOCK_STREAM, 0));

    struct sockaddr_un cli_addr;
    memset(&cli_addr, 0, sizeof(cli_addr));
    cli_addr.sun_family = AF_UNIX;
    throwOnError(sprintf(cli_addr.sun_path, "/tmp/shoz/%s-client-%d-%d.sock",
        this->metadata->topic_name, getpid(), gettid()));

    throwOnError(bind(sockfd, (struct sockaddr*) &cli_addr, sizeof(cli_addr)));

    struct sockaddr_un server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;
    throwOnError(sprintf(server_addr.sun_path, "/tmp/shoz/%s-server.sock",
        this->metadata->topic_name));

    // should wait until the server is ready
    struct stat buf;
    for (int tries = 0; tries < 2 && (stat(server_addr.sun_path, &buf) != 0); tries++) {
        usleep(50000);
    }
    throwOnError(connect(sockfd, (struct sockaddr*) &server_addr, sizeof(server_addr)));

    // prepare message
    struct msghdr msg;
    struct iovec iov[1];
    char ctrl_buf[CMSG_SPACE(sizeof(int))];
    char dummy[1];

    iov[0].iov_base = dummy;
    iov[0].iov_len = 1;

    msg.msg_name = nullptr;
    msg.msg_namelen = 0;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctrl_buf;
    msg.msg_controllen = sizeof(ctrl_buf);

    // receive the message with the shareable handle
    throwOnError(recvmsg(sockfd, &msg, 0));

    throwOnError(close(sockfd));
    throwOnError(unlink(cli_addr.sun_path));

    // import the shareable handle into a generic handle
    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_len != CMSG_LEN(sizeof(int))) throwError();
    
    ShareableHandle sh_handle = *((ShareableHandle*) CMSG_DATA(cmsg));
    throwOnErrorCuda(cuMemImportFromShareableHandle(&this->handle,
        (void*) (uintptr_t) sh_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    this->handle_is_valid = true;

    this->attachPool(true);
}

void ShareableAllocator::createPool(size_t size) {
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    size_t padded_size = this->getPaddedSize(size, &prop);
    throwOnErrorCuda(cuMemCreate(&this->handle, padded_size, &prop, 0));
    this->handle_is_valid = true;
    this->metadata->pool_size = padded_size;
}

void ShareableAllocator::attachPool(bool read_only) {
    CUdeviceptr dptr;
    CUmemAccessDesc acc_desc;
    size_t& size = this->metadata->pool_size;

    throwOnErrorCuda(cuMemAddressReserve(&dptr, size, 0, 0, 0));
    throwOnErrorCuda(cuMemMap(dptr, size, 0, this->handle, 0));

    acc_desc.location.id = 0;
    acc_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc_desc.flags = read_only ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    throwOnErrorCuda(cuMemSetAccess(dptr, size, &acc_desc, 1));

    this->pool_base = (void*) dptr;
}

void ShareableAllocator::detachPool() {
    if (!this->handle_is_valid) return;
    throwOnErrorCuda(cuMemRelease(this->handle));
    throwOnErrorCuda(cuMemUnmap((CUdeviceptr) this->pool_base, this->metadata->pool_size));
    throwOnErrorCuda(cuMemAddressFree((CUdeviceptr) this->pool_base, this->metadata->pool_size));
}

size_t ShareableAllocator::getPaddedSize(const size_t size, const CUmemAllocationProp* prop) const {
    size_t gran = 0;
    throwOnErrorCuda(cuMemGetAllocationGranularity(&gran, prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return ((size - 1) / gran + 1) * gran;
}