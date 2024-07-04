// #define ZENOHCXX_ZENOHC

#include <unistd.h>
#include <sys/time.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <iostream>

#include <cuda.h>
#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "examples/vector_arithmetic.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;

// #define PUB_NUM 1
// #define SUB_NUM 1

// #define VEC_SIZE 128

__global__ void __vecAdd(int* c, int* a, int* b) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

static inline double getMSec(timeval tv) {
    return (double) tv.tv_sec * 1000 + (double) tv.tv_usec / 1000;
}

struct timeval entryTime, beginTime, endTime;

int main(int argc, char *argv[]) {
    
    gettimeofday(&entryTime, 0);

    char *config_path = argv[1];

    Allocator::Domain domain = { Allocator::DeviceType::kGPU, 0};

    switch (fork()) {
    case -1:
        return -1;
    case 0: {
        /* Pub */ 
        try {
            cuInit(0);
            Publisher pub("topic 0", config_path, domain, 4096);

            int arr[1024];

            for (int i = 0; i < 1024; i++) arr[i] = rand() % 10;

            gettimeofday(&beginTime, 0);
            pub.put(arr, sizeof(int) * 1024);
            gettimeofday(&endTime, 0);

            cout << "beginTime: " << getMSec(beginTime) << ", endTime: " << getMSec(endTime) << endl;

        } catch (zenoh::ErrorMessage& err) {
            cerr << "Zenoh: " << err.as_string_view() << endl;
            exit(1);
        } catch (runtime_error& err) {
            cerr << "Publisher: " << err.what() << endl;
            exit(1);
        }
        break;
    }
    default: {
        /* Sub */
        try {
            cuInit(0);
            Subscriber sub("topic 0", config_path, domain, 4096);
            Subscriber::MessageHandler handler;

            int* c;
            cudaMalloc(&c, sizeof(int) * 512);

            handler = [c](void *msg) {
                gettimeofday(&beginTime, 0);
                int arr[512];
                int* a = (int*) msg;
                int* b = (int*) msg + 512;

                __vecAdd<<<1, 512>>>(c, a, b);

                cudaMemcpy(arr, c, sizeof(int) * 512, cudaMemcpyDeviceToHost);
                cout << "a + b:" << endl;
                for (int i = 0; i < 512; i++) cout << arr[i] << ' ';
                cout << endl;

                cout << "beginTime: " << getMSec(beginTime) << endl;
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
    }

    cout << "entry time: " << getMSec(entryTime) << endl;

    return 0;
}

