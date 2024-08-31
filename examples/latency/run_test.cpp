#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <zenoh.hxx>

#include "helpers.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;
using namespace hlp;

void pubTest(size_t size, size_t times);
void subTest(size_t size, size_t times);

const char kTopic[] = "latency-test";
const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
constexpr size_t kPoolSize = 2 * 1024 * 1024;  // 2 MiB

int main(int argc, char* argv[]) {
    if (argc < 2 || (stoi(argv[1]) == 0 && argc < 4)) {
        cerr << "Usage: " << argv[0] << " TYPE [SIZE] [TIMES]\n\n"
             << "TYPE:\n"
             << "  0    publisher\n"
             << "  1    subscriber\n"
             << "SIZE:\n"
             << "  size of the message to publish in bytes (only effective "
                "when TYPE=0)\n"
             << "TIMES:\n"
             << "  number of times to publish a message (only effective when "
                "TYPE=0)"
             << endl;
        exit(1);
    }
    int node_type = stoi(argv[1]);
    size_t size = stoul(argv[2]);
    size_t times = stoul(argv[3]);

    switch (node_type) {
    case 0:
        pubTest(size, times);
        break;
    case 1:
        subTest(size, times);
        break;
    default:
        cerr << "Unknown type" << endl;
        return 1;
    }

    return 0;
}

void pubTest(size_t size, size_t times) {
    cout << "size: " << size << endl;
    cout << "times: " << times << endl;

    Timer timer(10000);

    try {
        Publisher pub(kTopic, kDftLLocator, kPoolSize);

        char* arr = new char[size];
        for (int i = 0; i < size; i++) arr[i] = rand() % 10;

        for (int i = 0; i < times; i++) {
            timer.setPoint();
            pub.put(arr, size);
        }

        delete[] arr;
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }

    char filename[64];
    sprintf(filename, "pub-log-%lu-%lu.csv", size, times);
    timer.dump(filename);
}

void subTest(size_t size, size_t times) {
    Timer timer(10000);

    try {
        Subscriber sub(kTopic, kDftLLocator, kPoolSize);

        auto handler = [&timer](void* msg, size_t size) { timer.setPoint(); };

        sub.sub(handler);

        cout << "Type enter to leave" << endl;
        cin.get();
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }

    char filename[64];
    sprintf(filename, "sub-log-%lu-%lu.csv", size, times);
    timer.dump(filename);
}