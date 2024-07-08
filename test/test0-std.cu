// #define ZENOHCXX_ZENOHC

#include <unistd.h>
#include <sys/time.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <iostream>
#include <chrono>

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
StdTimeHelper timeHelper;
Allocator::Domain domain = { Allocator::DeviceType::kGPU, 0 };

void pubTest(char *config_path) {
    try {
        cuInit(0);
        Publisher pub("topic 0", config_path, domain, 4096);

        int arr[1024];

        for (int i = 0; i < 1024; i++) arr[i] = rand() % 10;

        timeHelper.setPoint();
        pub.put(arr, sizeof(int) * 1024);
        timeHelper.setPoint();

        auto durObj = timeHelper.getDuration<StdTimeHelper::millisec>(0, 1);

        cout << "publisher duration : " <<  durObj.count() << endl;

    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }
}

void subTest(char *config_path) {
    try {
        cuInit(0);
        Subscriber sub("topic 0", config_path, domain, 4096);
        Subscriber::MessageHandler handler;

        int* c;
        cudaMalloc(&c, sizeof(int) * 512);

        handler = [c](void *msg) {
            
            int arr[512];
            int* a = (int*) msg;
            int* b = (int*) msg + 512;

            __vecAdd<<<1, 512>>>(c, a, b);

            timeHelper.setPoint();
            cudaMemcpy(arr, c, sizeof(int) * 512, cudaMemcpyDeviceToHost);
            timeHelper.setPoint();

            cout << "a + b:" << endl;
            for (int i = 0; i < 512; i++) cout << arr[i] << ' ';
            cout << endl;

            cout << "beginTime: " << endl;
        };

        timeHelper.setPoint();
        sub.sub(handler);
        timeHelper.setPoint();
        
        sleep(5);
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }
}

int main(int argc, char *argv[]) {

    char *config_path = argv[1];

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

