#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>

#include <zenoh.hxx>

#include "node/publisher.hpp"
#include "node/subscriber.hpp"
#include "vector_arithmetic.hpp"

using namespace std;

void printUsageAndExit(char* program_name);
void runAsPublisher(const char* llocator, int n);
void runAsSubscriber(const char* llocator, int job);

const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
const char kTopicName[] = "cross_process";
const size_t kPoolSize = 1024;

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

    const char* llocator = optind < argc ? argv[optind] : kDftLLocator;
    if (run_as_pub) {
        runAsPublisher(llocator, n);
    } else {
        runAsSubscriber(llocator, job);
    }

    return 0;
}

void printUsageAndExit(char* program_name) {
    cerr << "Usage: " << program_name << " OPTIONS L_LOCATOR" << endl << endl;
    cerr << "OPTIONS:" << endl;
    cerr << "  -p N     run as a publisher" << endl;
    cerr << "     N     number of messages to publish" << endl;
    cerr << "  -s JOB   run as a subscriber" << endl;
    cerr << "     JOB=1 element-wise addition for 2 vectors" << endl;
    cerr << "     JOB=2 element-wise multiplication for 2 vectors" << endl;
    cerr << "     JOB=3 modify shared readonly memory" << endl << endl;
    cerr << "L_LOCATOR:" << endl;
    cerr << "  listening locator string for p2p mode (Default: "
            "udp/224.0.0.123:7447#iface=lo)"
         << endl;
    exit(1);
}

void runAsPublisher(const char* llocator, int n) {
    cout << "Run as a publisher" << endl;
    srand(time(nullptr) + getpid());

    try {
        Publisher publisher(kTopicName, llocator, kPoolSize);
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
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }
}

void runAsSubscriber(const char* llocator, int job) {
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
        Subscriber subscriber(kTopicName, llocator, kPoolSize);
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
        subscriber.sub(handler);

        cout << "Type enter to continue..." << endl;
        cin.get();
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }
}