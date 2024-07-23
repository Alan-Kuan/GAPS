#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <cuda.h>
#include <zenoh.hxx>

#include "helpers.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;
using namespace hlp;

void pubTest(size_t tsize, size_t times);
void subTest();

__global__ void __vecAdd(int* c, int* a, int* b) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

const size_t kPoolSize = 65536;
const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";

int main(int argc, char* argv[]) {
    if (argc < 2 || (argc < 4 && stoi(argv[1]) == 0)) {
        cerr << " Pub:\n";
        cerr << "     arg1 [number] 0\n";
        cerr << "     arg2 [number] size\n";
        cerr << "     arg3 [number] times\n";
        cerr << " Sub:\n";
        cerr << "     arg1 [number] 1\n";
        exit(1);
    }
    int node_type = stoi(argv[1]);

    switch (node_type) {
    case 0:
        pubTest(stoul(argv[2]), stoul(argv[3]));
        break;
    case 1:
        subTest();
        break;
    default:
        cerr << "not in case\n";
    }

    return 0;
}

void pubTest(size_t tsize, size_t times) {
    cout << "size: " << tsize << "\n";
    cout << "times: " << times << "\n";
    cout << "enter to send messages\n";
    cin.get();

    Domain domain = {DeviceType::kGPU, 0};
    size_t arr_size = tsize;
    size_t count = arr_size / sizeof(int);

    Timer timer(10000);
    try {
        cuInit(0);
        timer.setPoint();
        Publisher pub("topic0", kDftLLocator, domain, kPoolSize);
        timer.setPoint();

        int* arr = new int[count];

        for (int i = 0; i < count; i++) arr[i] = rand() % 10;

        for (int i = 0; i < times; i++) {
            cout << "time: " << i + 1 << "\n";
            timer.setPoint();
            pub.put(arr, arr_size);
            timer.setPoint();
        }

        delete[] arr;

    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }
    timer.writeAll("pub-log.csv");
}

void subTest() {
    Domain domain = {DeviceType::kGPU, 0};
    Timer timer(10000);

    try {
        cuInit(0);

        Subscriber sub("topic0", kDftLLocator, domain, kPoolSize);

        timer.setPoint();
        sub.sub([&timer](void* msg, size_t size) {
            timer.setPoint();

            size_t arr_size = size / 2;
            int count = arr_size / sizeof(int);
            int* arr = new int[count];
            int* a = (int*) msg;
            int* b = (int*) msg + count;
            int* c;
            cudaMalloc(&c, arr_size);

            timer.setPoint();
            __vecAdd<<<1, 512>>>(c, a, b);
            timer.setPoint();

            timer.setPoint();
            cudaMemcpy(arr, c, arr_size, cudaMemcpyDeviceToHost);
            timer.setPoint();

            cudaFree(c);

            delete[] arr;
        });
        timer.setPoint();
        cout << "enter to leave\n";
        cin.get();

    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }
    timer.writeAll("sub-log.csv");
}