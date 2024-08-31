#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <iceoryx_posh/runtime/posh_runtime.hpp>

#include "helpers.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"
#include "vector_arithmetic.hpp"

using namespace std;

void printUsageAndExit(char* program_name);
void runAsPublisher(int n);
void runAsSubscriber(int job);

const char kTopicName[] = "arithmetic";
const size_t kPoolSize = 2 * 1024 * 1024;  // 2 MiB

extern char* optarg;
extern int optind;

int main(int argc, char* argv[]) {
    if (argc < 3) printUsageAndExit(argv[0]);

    bool run_as_pub = false;
    bool run_as_sub = false;
    int n = 1;
    int job = 1;
    size_t end_idx = 0;
    int opt;

    try {
        while ((opt = getopt(argc, argv, "p:s:")) != -1) {
            switch (opt) {
            case 'p':
                run_as_pub = true;
                n = stoi(optarg, &end_idx, 10);
                if (optarg[end_idx] != '\0') {
                    cerr << "The given argument is not a number" << endl;
                    return 1;
                }
                break;
            case 's':
                run_as_sub = true;
                job = stoi(optarg, &end_idx, 10);
                if (optarg[end_idx] != '\0') {
                    cerr << "The given argument is not a number" << endl;
                    return 1;
                }
                break;
            default:
                printUsageAndExit(argv[0]);
            }
        }
    } catch (invalid_argument& err) {
        cerr << "The given argument is not a number" << endl;
        return 1;
    } catch (out_of_range& err) {
        cerr << "The given argument is too large or too small" << endl;
        return 1;
    }

    if (run_as_pub && run_as_sub) {
        cerr << "Should only be either a publisher or a subscriber" << endl;
        return 1;
    }
    if (n <= 0 || job < 1 || job > 3) printUsageAndExit(argv[0]);

    if (run_as_pub) {
        runAsPublisher(n);
    } else {
        runAsSubscriber(job);
    }

    return 0;
}

void printUsageAndExit(char* program_name) {
    cerr << "Usage: " << program_name << " OPTIONS" << endl << endl;
    cerr << "OPTIONS:" << endl;
    cerr << "  -p N     run as a publisher" << endl;
    cerr << "     N     number of messages to publish" << endl;
    cerr << "  -s JOB   run as a subscriber" << endl;
    cerr << "     JOB=1 element-wise addition for 2 vectors" << endl;
    cerr << "     JOB=2 element-wise multiplication for 2 vectors" << endl;
    cerr << "     JOB=3 modify shared readonly memory" << endl << endl;
    exit(1);
}

void runAsPublisher(int n) {
    cout << "Run as a publisher" << endl;
    srand(time(nullptr) + getpid());

    try {
        iox::runtime::PoshRuntime::initRuntime("cross_process_publisher");
        Publisher publisher(kTopicName, kPoolSize);
        int arr[128];

        for (int T = 0; T < n; T++) {
            cout << '#' << (T + 1) << ":" << endl;
            for (int i = 0; i < 128; i++) arr[i] = rand() % 10;
            cout << "  A: ";
            for (int i = 0; i < 64; i++) cout << arr[i] << ' ';
            cout << endl << "  B: ";
            for (int i = 64; i < 128; i++) cout << arr[i] << ' ';
            cout << endl;

            publisher.put(arr, sizeof(int) * 128);
        }
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }
}

void runAsSubscriber(int job) {
    switch (job) {
    case 1:
        cout << "Run as a subscriber (element-wise addition)" << endl;
        break;
    case 2:
        cout << "Run as a subscriber (element-wise multiplication)" << endl;
        break;
    case 3:
        cout << "Run as a naughty subscriber (write to read-only memory)"
             << endl;
        break;
    }

    try {
        iox::runtime::PoshRuntime::initRuntime("cross_process_subscriber");
        Subscriber::MessageHandler handler;

        switch (job) {
        case 1:
            handler = [](void* msg, size_t size) {
                size_t arr_size = size / 2;
                int count = arr_size / sizeof(int);
                int* arr = new int[count];
                int* a = (int*) msg;
                int* b = (int*) msg + count;
                int* c;
                cudaMalloc(&c, arr_size);

                vecAdd(c, a, b, count);
                cudaMemcpy(arr, c, arr_size, cudaMemcpyDeviceToHost);
                cudaFree(c);

                cout << "a + b:" << endl;
                for (int i = 0; i < count; i++) cout << arr[i] << ' ';
                cout << endl;
                delete[] arr;
            };
            break;
        case 2:
            handler = [](void* msg, size_t size) {
                size_t arr_size = size / 2;
                int count = arr_size / sizeof(int);
                int* arr = new int[count];
                int* a = (int*) msg;
                int* b = (int*) msg + count;
                int* c;
                cudaMalloc(&c, arr_size);

                vecMul(c, a, b, count);
                cudaMemcpy(arr, c, arr_size, cudaMemcpyDeviceToHost);
                cudaFree(c);

                cout << "a x b:" << endl;
                for (int i = 0; i < 64; i++) cout << arr[i] << ' ';
                cout << endl;
                delete[] arr;
            };
            break;
        case 3:
            handler = [](void* msg, size_t _) {
                int* a = (int*) msg;

                cout << cudaGetErrorString(cudaMemset(a, 0, sizeof(int) * 64))
                     << endl;
            };
            break;
        }
        Subscriber subscriber(kTopicName, kPoolSize, handler);

        cout << "Ctrl+C to leave" << endl;
        hlp::waitForSigInt();
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }
}