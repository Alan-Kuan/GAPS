#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <iostream>

#include <cuda.h>
#include <zenoh.hxx>

#include "allocator/allocator.hpp"
#include "error.hpp"
#include "examples/vector_arithmetic.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;

void printUsageAndExit(char* program_name);
void runAsPublisher(const char* conf_path);
void runAsSubscriber(const char* conf_path, int job);

const char kTopicName[] = "cross_process";
const size_t kPoolSize = 1024;

extern char* optarg;
extern int optind;

int main(int argc, char* argv[]) {
    if (argc < 3) printUsageAndExit(argv[0]);

    bool run_as_pub = false;
    int job = 0;

    int opt;
    while ((opt = getopt(argc, argv, "ps:")) != -1) {
        switch (opt) {
        case 'p':
            run_as_pub = true;
            break;
        case 's':
            job = strtol(optarg, nullptr, 10);
            break;
        default:
            printUsageAndExit(argv[0]);
        }
    }

    if (run_as_pub) {
        runAsPublisher(argv[optind]);
    } else {
        runAsSubscriber(argv[optind], job);
    }

    return 0;
}

void printUsageAndExit(char* program_name) {
    cerr << "Usage: " << program_name << " OPTIONS ZENOH_CONFIG" << endl << endl;
    cerr << "OPTIONS:" << endl;
    cerr << "  -p       run as a publisher" << endl;
    cerr << "  -s JOB   run as a subscriber" << endl;
    cerr << "     1     element-wise addition for 2 vectors" << endl;
    cerr << "     2     element-wise multiplication for 2 vectors" << endl;
    cerr << "     3     modify shared readonly memory" << endl;
    exit(1);
}

void runAsPublisher(const char* conf_path) {
    cout << "Run as a publisher" << endl;
    srand(time(nullptr));

    try {
        if (cuInit(0) != CUDA_SUCCESS) throwError();

        Allocator::Domain domain = { Allocator::DeviceType::kGPU, 0 };
        Publisher publisher(kTopicName, conf_path, domain, kPoolSize);
        int arr[128];

        for (int i = 0; i < 128; i++) arr[i] = rand() % 10;
        cout << "A:" << endl;
        for (int i = 0; i < 64; i++) cout << arr[i] << ' ';
        cout << endl << "B:" << endl;
        for (int i = 64; i < 128; i++) cout << arr[i] << ' ';
        cout << endl;

        publisher.put(arr, sizeof(int) * 128);
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }
}

void runAsSubscriber(const char* conf_path, int job) {
    switch (job) {
    case 1:
        cout << "Run as a subscriber (element-wise addition)" << endl;
        break;
    case 2:
        cout << "Run as a subscriber (element-wise multiplication)" << endl;
        break;
    case 3:
        cout << "Run as a naughty subscriber (write to read-only memory)" << endl;
        break;
    }

    try {
        if (cuInit(0) != CUDA_SUCCESS) throwError();

        Allocator::Domain domain = { Allocator::DeviceType::kGPU, 0 };
        Subscriber subscriber(kTopicName, conf_path, domain, kPoolSize);
        Subscriber::MessageHandler handler;

        int* c;
        cudaMalloc(&c, sizeof(int) * 64);

        switch (job) {
        case 1:
            handler = [c](void* msg) {
                int arr[64];
                int* a = (int*) msg;
                int* b = (int*) msg + 64;

                vecAdd(c, a, b, 64);
                cudaMemcpy(arr, c, sizeof(int) * 64, cudaMemcpyDeviceToHost);
                cout << "a + b:" << endl;
                for (int i = 0; i < 64; i++) cout << arr[i] << ' ';
                cout << endl;
            };
            break;
        case 2:
            handler = [c](void* msg) {
                int arr[64];
                int* a = (int*) msg;
                int* b = (int*) msg + 64;

                vecMul(c, a, b, 64);
                cudaMemcpy(arr, c, sizeof(int) * 64, cudaMemcpyDeviceToHost);
                cout << "a x b:" << endl;
                for (int i = 0; i < 64; i++) cout << arr[i] << ' ';
                cout << endl;
            };
            break;
        case 3:
            handler = [](void* msg) {
                int* a = (int*) msg;

                cout << cudaGetErrorString(cudaMemset(a, 0, sizeof(int) * 64)) << endl;
            };
            break;
        }
        subscriber.sub(handler);

        cout << "Type enter to continue..." << endl;
        getchar();
        cudaFree(c);
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }
}