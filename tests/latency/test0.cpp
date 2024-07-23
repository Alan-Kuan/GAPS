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

void pubTest(const char* llocator, size_t pool_size);
void subTest(const char* llocator, size_t pool_size);

/* Global */
const char kTopic[] = "topic 0";
const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
const size_t kPoolSize = 65536;

Domain domain = {DeviceType::kGPU, 0};
sem_t* sem_ready;
size_t transmit_size = 0;
size_t asize = 0;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " [size] (4 ~ " << kPoolSize << ")"
             << endl;
        exit(1);
    }

    sem_ready = sem_open("/test0_sem_ready", O_CREAT, 0660, 0);
    transmit_size = stoul(argv[1]);
    asize = transmit_size / sizeof(int);

    switch (fork()) {
    case -1:
        return -1;
    case 0:
        pubTest(kDftLLocator, kPoolSize);
        break;
    default:
        subTest(kDftLLocator, kPoolSize);
    }

    sem_close(sem_ready);
    sem_unlink("/test0_sem_ready");
    return 0;
}

void pubTest(const char* llocator, size_t pool_size) {
    Timer timer;

    try {
        cuInit(0);

        timer.setPoint();
        Publisher pub(kTopic, llocator, domain, pool_size);
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

    timer.writeAll("pub-log-test0.csv");
}

void subTest(const char* llocator, size_t pool_size) {
    Timer timer;
    sem_t sem_subend;
    sem_init(&sem_subend, 0, 0);

    try {
        cuInit(0);
        timer.setPoint();
        Subscriber sub(kTopic, llocator, domain, pool_size);
        timer.setPoint();

        auto handler = [&sem_subend, &timer](void* msg, size_t size) {
            timer.setPoint();
            sem_post(&sem_subend);
        };

        timer.setPoint();
        sub.sub(handler);
        timer.setPoint();
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
    timer.showAll("sub-log-test0.csv");
}