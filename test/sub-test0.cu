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
#include "helpers.hpp"

using namespace std;
using namespace hlp;

__global__ void __vecAdd(int* c, int* a, int* b) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

/* Global */
TimeHelper entryTime;
TimeHelper beginTime;
TimeHelper endTime;
Allocator::Domain domain = { Allocator::DeviceType::kGPU, 0 };

void subTest(char *config_path) {
    try {
        cuInit(0);
        Subscriber sub("topic 0", config_path, domain, 4096);
        Subscriber::MessageHandler handler;

        int* c;
        cudaMalloc(&c, sizeof(int) * 512);

        handler = [c](void *msg) {
            beginTime.setPoint();
            int arr[512];
            int* a = (int*) msg;
            int* b = (int*) msg + 512;

            __vecAdd<<<1, 512>>>(c, a, b);

            cudaMemcpy(arr, c, sizeof(int) * 512, cudaMemcpyDeviceToHost);
            cout << "a + b:" << endl;
            for (int i = 0; i < 512; i++) cout << arr[i] << ' ';
            cout << endl;

            cout << "beginTime: " << beginTime.getMSec() << endl;
        };

        sub.sub(handler);
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }
}

int main(int argc, char *argv[]) {

    entryTime.setPoint();
    char *config_path = argv[1];

    subTest(config_path);

    cout << argv[0] << " entry time: " << entryTime.getMSec() << endl;

    return 0;
}

