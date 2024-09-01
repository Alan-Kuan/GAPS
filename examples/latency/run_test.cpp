#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <zenoh.hxx>

#include "helpers.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;
using namespace hlp;

void pubTest(int nproc, const char* output_name, size_t size, size_t times);
void subTest(int nproc, const char* output_name);

const char kTopic[] = "latency-test";
const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
constexpr size_t kPoolSize = 2 * 1024 * 1024;  // 2 MiB

int main(int argc, char* argv[]) {
    if (argc < 2 || (stoi(argv[1]) == 0 && argc < 4)) {
        cerr << "Usage: " << argv[0] << " TYPE NPROC OUTPUT [SIZE] [TIMES]\n\n"
             << "TYPE:\n"
             << "  0    publisher\n"
             << "  1    subscriber\n"
             << "NPROC:\n"
             << "  number of publishers / subscribers\n"
             << "OUTPUT:\n"
             << "  name of the output csv\n"
             << "SIZE:\n"
             << "  size of the message to publish in bytes (only required "
                "when TYPE=0)\n"
             << "TIMES:\n"
             << "  number of times to publish a message (only required when "
                "TYPE=0)"
             << endl;
        exit(1);
    }
    int node_type = stoi(argv[1]);
    int nproc = stoi(argv[2]);
    const char* output_name = argv[3];
    size_t size, times;

    switch (node_type) {
    case 0:
        size = stoul(argv[4]);
        times = stoul(argv[5]);
        pubTest(nproc, output_name, size, times);
        break;
    case 1:
        subTest(nproc, output_name);
        break;
    default:
        cerr << "Unknown type" << endl;
        return 1;
    }

    return 0;
}

void pubTest(int nproc, const char* output_name, size_t size, size_t times) {
    cout << "size: " << size << endl;
    cout << "times: " << times << endl;

    pid_t pid = -1;
    int p = 1;

    for (; p < nproc; p++) {
        pid = fork();
        if (pid < 0) {
            cerr << "Failed to fork" << endl;
            exit(1);
        } else if (pid == 0) {
            break;
        }
    }

    Timer timer(times);

    try {
        Publisher pub(kTopic, kDftLLocator, kPoolSize);
        int count = size / sizeof(int);
        int* arr = new int[count];

        for (int t = 0; t < times; t++) {
            int tag = (p - 1) * times + t;
            memset(arr, rand() % 256, size);
            arr[0] = tag;

            timer.setPoint(tag);
            pub.put(arr, size);
            // another time point is set at the subscriber-end

            // NOTE: iceoryx subscriber may miss messages if publishes too
            // frequently
            usleep(50000);  // 50ms
        }

        delete[] arr;
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }

    stringstream ss;
    ss << "pub-" << output_name << '-' << p << ".csv";
    timer.dump(ss.str().c_str());

    if (pid == 0) return;
    for (int i = 1; i < nproc; i++) wait(nullptr);
}

void subTest(int nproc, const char* output_name) {
    pid_t pid = -1;
    int p = 1;

    for (; p < nproc; p++) {
        pid = fork();
        if (pid < 0) {
            cerr << "Failed to fork" << endl;
            exit(1);
        } else if (pid == 0) {
            break;
        }
    }

    Timer timer(10000);

    try {
        Subscriber sub(kTopic, kDftLLocator, kPoolSize);
        auto handler = [&timer](void* msg, size_t size) {
            // upon received, get the current time point
            auto now = timer.now();

            // NOTE: though the tag has been copied to the memory synchronously
            // before publishing, for some reason, if we read it too quickly, we
            // will read old data
            usleep(1000);

            // find the tag from the message
            int* buf = (int*) malloc(size);
            cudaMemcpy(buf, msg, size, cudaMemcpyDeviceToHost);
            timer.setPoint(std::move(now), buf[0]);
            cout << "Saved point " << buf[0] << endl;
            free(buf);
        };
        sub.sub(handler);

        if (pid != 0) cout << "Ctrl+C to leave" << endl;
        hlp::waitForSigInt();
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }

    stringstream ss;
    ss << "sub-" << output_name << '-' << p << ".csv";
    timer.dump(ss.str().c_str());

    if (pid == 0) return;
    for (int i = 1; i < nproc; i++) wait(nullptr);
}