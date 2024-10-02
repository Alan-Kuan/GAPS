#include <sys/wait.h>

#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <cuda_runtime.h>
#include <iceoryx_posh/runtime/posh_runtime.hpp>

#include "helpers.hpp"
#include "init.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;
using namespace hlp;
using namespace chrono_literals;

void pubTest(int nproc, const char* output_name, size_t size, size_t times);
void subTest(int nproc, const char* output_name);

const char kTopic[] = "latency-test";
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
             << "  size of the message to publish in bytes (only required when "
                "TYPE=0)\n"
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
        char runtime_name[32];
        sprintf(runtime_name, "latency_test_publisher_%d", p);
        iox::runtime::PoshRuntime::initRuntime(runtime_name);
        Publisher pub(kTopic, kPoolSize);

        for (int t = 0; t < times; t++) {
            int tag = (p - 1) * times + t + 1;
            int* buf_d;
            do {
                buf_d = (int*) pub.malloc(size);
            } while (!buf_d);
            init_data(buf_d, size / sizeof(int), tag);

            timer.setPoint(tag);
            // since we won't use the data from the subscriber side, it's ok to
            // exploit the size field to send the tag
            pub.put(buf_d, (size_t) tag);
            // another time point is set at the subscriber-end

            // control publishing frequency
            this_thread::sleep_for(1ms);
        }

        if (pid != 0) cout << "Ctrl+C to leave" << endl;
        hlp::waitForSigInt();
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
        char runtime_name[32];
        sprintf(runtime_name, "latency_test_subscriber_%d", p);
        iox::runtime::PoshRuntime::initRuntime(runtime_name);
        auto handler = [&timer](void* msg, size_t tag) {
            // upon received, set the current time point
            timer.setPoint((int) tag);
        };
        Subscriber sub(kTopic, kPoolSize, handler);

        if (pid != 0) cout << "Ctrl+C to leave" << endl;
        hlp::waitForSigInt();
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