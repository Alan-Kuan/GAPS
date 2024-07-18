#include <fcntl.h>
#include <semaphore.h>
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
#include "helpers.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;
using namespace hlp;

__global__ void __vecAdd(int* c, int* a, int* b) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

/* Global */
#define POOL_SIZE 65536
const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
Domain domain = {DeviceType::kGPU, 0};
sem_t* sem_ready;
size_t transmit_size = 0;
size_t asize = 0;

void pubTest(const char* zenConfig) {
    Timer timer;
    try {
        cuInit(0);
        timer.setPoint();
        Publisher pub("topic 0", zenConfig, domain, POOL_SIZE);
        timer.setPoint();

        int* arr = new int[asize];

        for (int i = 0; i < asize; i++) arr[i] = rand() % 10;

        sem_wait(sem_ready);
        for (int i = 0; i < 5; i++) {
            timer.setPoint();
            pub.put(arr, transmit_size);
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
    timer.showAll("pub");
}

void subTest(const char* zenConfig) {
    Timer timer;
    sem_t sem_subend;
    sem_init(&sem_subend, 0, 0);
    try {
        cuInit(0);
        Subscriber sub("topic 0", zenConfig, domain, POOL_SIZE);
        Subscriber::MessageHandler handler;

        int* c;
        size_t vsize = asize / 2;  // decompose data from publisher
        cudaMalloc(&c, sizeof(int) * vsize);

        handler = [c, &sem_subend, &timer](void* msg, size_t size) {
            timer.setPoint();
            size_t arr_size = size / 2;
            size_t count = arr_size / sizeof(int);
            int* arr = new int[count];
            int* a = (int*) msg;
            int* b = (int*) msg + count;

            timer.setPoint();
            __vecAdd<<<1, 512>>>(c, a, b);
            timer.setPoint();

            timer.setPoint();
            cudaMemcpy(arr, c, arr_size, cudaMemcpyDeviceToHost);
            timer.setPoint();
            // cout << "a + b:" << endl;
            // for (int i = 0; i < 512; i++) cout << arr[i] << ' ';
            // cout << endl;
            sem_post(&sem_subend);
        };

        timer.setPoint();
        sub.sub(handler);
        timer.setPoint();
        // usleep(1000000);
        sem_post(sem_ready);

        sem_wait(&sem_subend);

    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }

    sem_close(&sem_subend);
    timer.showAll("sub");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "usage: ./test0 [size](4~" << POOL_SIZE << ")\n";
        exit(1);
    }

    const char* config = kDftLLocator;

    sem_ready = sem_open("/sem_share", O_CREAT, 0660, 0);
    transmit_size = stoul(argv[1]);
    asize = transmit_size / sizeof(int);

    switch (fork()) {
    case -1:
        return -1;
    case 0:
        pubTest(config);
        break;
    default:
        subTest(config);
    }

    sem_close(sem_ready);
    sem_unlink("/sem_share");
    return 0;
}
