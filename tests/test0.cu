// #define ZENOHCXX_ZENOHC

#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>

#include <cuda.h>
#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "examples/vector_arithmetic.hpp"
#include "helpers.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;
using namespace hlp;

__global__ void __vecAdd(int* c, int* a, int* b) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

/* Global */
TimePoint beginTime;
TimePoint endTime;
Domain domain = {DeviceType::kGPU, 0};

void pubTest(char* config_path) {
    try {
        cuInit(0);
        Publisher pub("topic 0", config_path, domain, 4096);

        int arr[1024];

        for (int i = 0; i < 1024; i++) arr[i] = rand() % 10;

        beginTime.set();
        pub.put(arr, sizeof(int) * 1024);
        endTime.set();

        sleep(1);

        cout << "pub begin: " << fixed << beginTime.getMSec();
        cout << ", end: " << fixed << endTime.getMSec() << endl;

    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }
}

void subTest(char* config_path) {
    try {
        cuInit(0);
        Subscriber sub("topic 0", config_path, domain, 4096);
        Subscriber::MessageHandler handler;

        int* c;
        cudaMalloc(&c, sizeof(int) * 512);

        handler = [c](void* msg, size_t size) {
            beginTime.set();
            int arr[512];
            int* a = (int*) msg;
            int* b = (int*) msg + 512;

            __vecAdd<<<1, 512>>>(c, a, b);

            cudaMemcpy(arr, c, sizeof(int) * 512, cudaMemcpyDeviceToHost);
            // cout << "a + b:" << endl;
            // for (int i = 0; i < 512; i++) cout << arr[i] << ' ';
            // cout << endl;

            cout << "sub handle begin: " << fixed << beginTime.getMSec()
                 << endl;
        };

        sub.sub(handler);
        sleep(5);
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    char* config_path = argv[1];

    switch (fork()) {
    case -1:
        return -1;
    case 0:
        pubTest(config_path);
        break;
    default:
        subTest(config_path);
    }

    return 0;
}
