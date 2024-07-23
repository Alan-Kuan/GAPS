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

const char kTopic[] = "test0-p2p";
const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
const size_t kPoolSize = 65536;

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

    switch (node_type) {
    case 0:
        pubTest(stoul(argv[2]), stoul(argv[3]));
        break;
    case 1:
        subTest();
        break;
    default:
        cerr << "Unknown type" << endl;
        return 1;
    }

    return 0;
}

void pubTest(size_t tsize, size_t times) {
    cout << "size: " << tsize << endl;
    cout << "times: " << times << endl;
    cout << "Type enter to send messages" << endl;
    cin.get();

    Timer timer(10000);
    Domain domain = {DeviceType::kGPU, 0};
    size_t arr_size = tsize;
    size_t count = arr_size / sizeof(int);

    try {
        cuInit(0);

        timer.setPoint();
        Publisher pub(kTopic, kDftLLocator, domain, kPoolSize);
        timer.setPoint();

        int* arr = new int[count];
        for (int i = 0; i < count; i++) arr[i] = rand() % 10;

        for (int i = 0; i < times; i++) {
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

    timer.dump("pub-log-test1.csv");
}

void subTest() {
    Timer timer(10000);
    Domain domain = {DeviceType::kGPU, 0};

    try {
        cuInit(0);

        timer.setPoint();
        Subscriber sub(kTopic, kDftLLocator, domain, kPoolSize);
        timer.setPoint();

        auto handler = [&timer](void* msg, size_t size) { timer.setPoint(); };

        timer.setPoint();
        sub.sub(handler);
        timer.setPoint();

        cout << "Type enter to leave" << endl;
        cin.get();
    } catch (zenoh::ErrorMessage& err) {
        cerr << "Zenoh: " << err.as_string_view() << endl;
        exit(1);
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }

    timer.dump("sub-log-test1.csv");
}