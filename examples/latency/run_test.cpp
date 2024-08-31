#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <iceoryx_posh/runtime/posh_runtime.hpp>

#include "helpers.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;
using namespace hlp;

void pubTest(const char* output_name, size_t size, size_t times);
void subTest(const char* output_name);

const char kTopic[] = "latency-test";
constexpr size_t kPoolSize = 2 * 1024 * 1024;  // 2 MiB

int main(int argc, char* argv[]) {
    if (argc < 2 || (stoi(argv[1]) == 0 && argc < 4)) {
        cerr << "Usage: " << argv[0] << " TYPE OUTPUT [SIZE] [TIMES]\n\n"
             << "TYPE:\n"
             << "  0    publisher\n"
             << "  1    subscriber\n"
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
    const char* output_name = argv[2];
    size_t size, times;

    switch (node_type) {
    case 0:
        size = stoul(argv[3]);
        times = stoul(argv[4]);
        pubTest(output_name, size, times);
        break;
    case 1:
        subTest(output_name);
        break;
    default:
        cerr << "Unknown type" << endl;
        return 1;
    }

    return 0;
}

void pubTest(const char* output_name, size_t size, size_t times) {
    cout << "size: " << size << endl;
    cout << "times: " << times << endl;

    Timer timer(10000);

    try {
        iox::runtime::PoshRuntime::initRuntime("latency_test_publisher");
        Publisher pub(kTopic, kPoolSize);

        char* arr = new char[size];
        for (int i = 0; i < size; i++) arr[i] = rand() % 10;

        for (int i = 0; i < times; i++) {
            timer.setPoint();
            pub.put(arr, size);
            // NOTE: iceoryx subscriber may miss messages if publishes too
            // frequently
            usleep(50000);  // 50ms
        }

        delete[] arr;
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }

    stringstream ss;
    ss << "pub-" << output_name << ".csv";
    timer.dump(ss.str().c_str());
}

void subTest(const char* output_name) {
    Timer timer(10000);

    try {
        iox::runtime::PoshRuntime::initRuntime("cross_process_subscriber");
        auto handler = [&timer](void* msg, size_t size) { timer.setPoint(); };
        Subscriber sub(kTopic, kPoolSize, handler);

        cout << "Ctrl+C to leave" << endl;
        hlp::waitForSigInt();
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }

    stringstream ss;
    ss << "sub-" << output_name << ".csv";
    timer.dump(ss.str().c_str());
}