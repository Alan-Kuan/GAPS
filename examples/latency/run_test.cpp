#include <sys/wait.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <zenoh.hxx>

#include "helpers.hpp"
#include "init.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;
namespace z = zenoh;

void pubTest(int nproc, const char* output_name, size_t size, size_t times);
void subTest(int nproc, const char* output_name);

const char kTopic[] = "latency-test";
const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
const auto kPubInterval = 20ms;
constexpr size_t kPoolSize = 32 * 1024 * 1024;  // 32 MiB

int main(int argc, char* argv[]) {
    if (argc < 4 || (argv[1][0] == 'p' && argc < 6)) {
        cerr << "Usage: " << argv[0] << " TYPE NPROC OUTPUT [SIZE] [TIMES]\n\n"
             << "TYPE:\n"
             << "  p    publisher\n"
             << "  s    subscriber\n"
             << "NPROC:\n"
             << "  number of publishers / subscribers\n"
             << "OUTPUT:\n"
             << "  prefix of the output csv (may contain directory)\n"
             << "SIZE:\n"
             << "  size of the message to publish in bytes (only required when "
                "TYPE=p)\n"
             << "TIMES:\n"
             << "  number of times to publish a message (only required when "
                "TYPE=p)"
             << endl;
        return 1;
    }
    char node_type = argv[1][0];
    int nproc = stoi(argv[2]);
    const char* output_name = argv[3];
    size_t size;
    int times;

#ifndef NDEBUG
    cout << "The code was compiled without NDEBUG macro defined, which may "
            "contain codes that influence time measurement."
         << endl;
#endif

    switch (node_type) {
    case 'p':
        size = stoul(argv[4]);
        times = stoi(argv[5]);
        pubTest(nproc, output_name, size, times);
        break;
    case 's':
        subTest(nproc, output_name);
        break;
    default:
        cerr << "Unknown type" << endl;
        return 1;
    }

    return 0;
}

void pubTest(int nproc, const char* output_name, size_t size, size_t times) {
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

    hlp::Timer timer(times);

    try {
        auto config = z::Config::create_default();
        config.insert(Z_CONFIG_MODE_KEY, Z_CONFIG_MODE_PEER);
        config.insert(Z_CONFIG_LISTEN_KEY, kDftLLocator);
        z::Session session(std::move(config));
        Publisher pub(session, kTopic, kPoolSize);

        for (int t = 0; t < times; t++) {
            int tag = (p - 1) * times + t + 1;
            int* buf_d;
            do {
                buf_d = (int*) pub.malloc(size);
            } while (!buf_d);
            init::fillArray(buf_d, size / sizeof(int), tag);

            timer.setPoint(tag);
            // since we won't use the data from the subscriber side, it's ok to
            // exploit the size field to send the tag
            pub.put(buf_d, (size_t) tag);
            // another time point is set at the subscriber-end

            // control publishing frequency
            this_thread::sleep_for(kPubInterval);
        }
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }

    stringstream ss;
    ss << output_name << "-" << p << ".csv";
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

    hlp::Timer timer(10000);

    try {
        auto config = z::Config::create_default();
        config.insert(Z_CONFIG_MODE_KEY, Z_CONFIG_MODE_PEER);
        config.insert(Z_CONFIG_LISTEN_KEY, kDftLLocator);
        z::Session session(std::move(config));
        Subscriber sub(session, kTopic, kPoolSize,
                       [&timer](void* msg, size_t tag) {
                           // upon received, set the current time point
                           timer.setPoint(tag);
                       });

        if (pid != 0) cout << "Ctrl+C to leave" << endl;
        hlp::waitForSigInt();
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }

    stringstream ss;
    ss << output_name << "-" << p << ".csv";
    timer.dump(ss.str().c_str());

    if (pid == 0) return;
    for (int i = 1; i < nproc; i++) wait(nullptr);
}