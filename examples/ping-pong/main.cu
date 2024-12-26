#include <unistd.h>

#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <curand_kernel.h>
#include <iceoryx_posh/runtime/posh_runtime.hpp>

#include "node/publisher.hpp"
#include "node/subscriber.hpp"
#include "rand_init.hpp"

/**
 *  Ping Pong Test:
 *   1. randomly generate two integers a and b
 *   2. ping side publishes a randomly generated vector v to the pong side
 *   3. pong side publishes back the result of a*v + b
 *   4. ping side also calculates a*v + b and compare with the received one
 */

using namespace std;

const char kTopicNamePing[] = "pp-ping";
const char kTopicNamePong[] = "pp-pong";
const size_t kPoolSize = 2 * 1024 * 1024;  // 2 MiB
const size_t kBufSize = 1024;              // 1 KiB
const size_t kBufCount = kBufSize / sizeof(int);
const int kTotalTimes = 5;

void runAsPingSide(int a, int b);
void runAsPongSide(int a, int b);

int main() {
    srand(time(nullptr));

    int a = rand() % 1000;
    int b = rand() % 1000;

    switch (fork()) {
    case -1:
        cerr << "Failed to fork" << endl;
        return 1;
    case 0:
        runAsPongSide(a, b);
        return 0;
    default:
        runAsPingSide(a, b);
    }

    return 0;
}

__global__ void __vecTransform(int* vec_o, int* vec_i, int a, int b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kBufCount) return;
    vec_o[idx] = a * vec_i[idx] + b;
}

void vecTransform(int* vec_o, int* vec_i, int a, int b) {
    int grid_dim = 1;
    int block_dim = kBufCount;
    if (kBufCount > 1024) {
        grid_dim = ((kBufCount - 1) >> 10) + 1;
        block_dim = 1024;
    }
    __vecTransform<<<grid_dim, block_dim>>>(vec_o, vec_i, a, b);
    cudaDeviceSynchronize();
}

void runAsPingSide(int a, int b) {
    // init random state
    curandState* states;
    cudaMalloc(&states, sizeof(curandState) * kBufCount);
    initRandStates(states, kBufCount, time(nullptr));

    int* res_d;
    cudaMalloc(&res_d, kBufSize);

    int* res = (int*) std::malloc(kBufSize);
    int* data = (int*) std::malloc(kBufSize);

    try {
        mutex cv_m;
        condition_variable cv;

        char runtime_name[32];
        sprintf(runtime_name, "ping-pong-publisher");
        iox::runtime::PoshRuntime::initRuntime(runtime_name);
        Publisher publisher(kTopicNamePing, kPoolSize);

        auto handler = [data, res, &cv](void* data_d, size_t size) {
            cudaMemcpy(data, data_d, kBufSize, cudaMemcpyDeviceToHost);
            if (memcmp(res, data, kBufSize) == 0) {
                cout << "Passed!" << endl;
            } else {
                cout << "Failed!" << endl;
            }
            cv.notify_one();
        };
        Subscriber subscriber(kTopicNamePong, kPoolSize, handler);

        // make sure both sides are ready
        cout << "Ready..." << endl;
        this_thread::sleep_for(2s);
        cout << "Start!" << endl;

        unique_lock lock{cv_m};
        for (int i = 0; i < kTotalTimes; i++) {
            // generate random vector
            int* vec_d = (int*) publisher.malloc(kBufSize);
            fillRandVals(states, vec_d, kBufCount);
            vecTransform(res_d, vec_d, a, b);
            cudaMemcpy(res, res_d, kBufSize, cudaMemcpyDeviceToHost);

            // publish the random vector
            publisher.put(vec_d, kBufSize);
            cout << "- - -" << endl;
            cout << "Ping!" << endl;
            cv.wait_for(lock, 1s);
        }
        lock.unlock();
    } catch (runtime_error& err) {
        cerr << "Ping Side: " << err.what() << endl;
        exit(1);
    }

    std::free(res);
    std::free(data);
    cudaFree(res_d);
    cudaFree(states);
}

void runAsPongSide(int a, int b) {
    try {
        mutex cv_m;
        condition_variable cv;
        int times = 0;

        char runtime_name[32];
        sprintf(runtime_name, "ping-pong-subscriber");
        iox::runtime::PoshRuntime::initRuntime(runtime_name);
        Publisher publisher(kTopicNamePong, kPoolSize);
        int* res_d = (int*) publisher.malloc(kBufSize);

        auto handler = [&publisher, res_d, a, b, &cv, &times](void* data,
                                                              size_t size) {
            vecTransform(res_d, (int*) data, a, b);
            publisher.put(res_d, kBufSize);
            cout << "Pong!" << endl;
            times++;
            cv.notify_one();
        };
        Subscriber subscriber(kTopicNamePing, kPoolSize, handler);

        unique_lock lock{cv_m};
        cv.wait(lock, [&times] { return times == kTotalTimes; });
        lock.unlock();
    } catch (runtime_error& err) {
        cerr << "Pong Side: " << err.what() << endl;
        exit(1);
    }
}