#include <sys/wait.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <iceoryx_hoofs/log/logmanager.hpp>
#include <iceoryx_posh/runtime/posh_runtime.hpp>

#include "env.hpp"
#include "helpers.hpp"
#include "init.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"

using namespace std;

void runAsPublisher(int id, const char* output_name, size_t payload_size);
void runAsSubscriber(int id, const char* output_name);

int main(int argc, char* argv[]) {
    if (argc < 4 || (argv[1][0] == 'p' && argc < 5)) {
        cerr << "Usage: " << argv[0] << " TYPE NPROC OUTPUT [SIZE]\n\n"
             << "TYPE:\n"
             << "  p    publisher\n"
             << "  s    subscriber\n"
             << "NPROC:\n"
             << "  number of publishers / subscribers\n"
             << "OUTPUT:\n"
             << "  prefix of the output csv (may contain directory)\n"
             << "SIZE:\n"
             << "  size of the message to publish in bytes (only required when "
                "TYPE=p)"
             << endl;
        return 1;
    }
    char node_type = argv[1][0];
    int nproc = stoi(argv[2]);
    const char* output_name = argv[3];
    size_t payload_size;

#ifndef NDEBUG
    cout << "The code was compiled without NDEBUG macro defined, which may "
            "contain codes that influence time measurement."
         << endl;
#endif

    pid_t pid = -1;
    int id = 1;

    for (; id < nproc; id++) {
        pid = fork();
        if (pid < 0) {
            cerr << "Failed to fork" << endl;
            exit(1);
        } else if (pid == 0) {
            break;
        }
    }

    iox::log::LogManager::GetLogManager().SetDefaultLogLevel(
        iox::log::LogLevel::kOff);

    char runtime_name[32];
    sprintf(runtime_name, "latency_test_%c", node_type);
    iox::runtime::PoshRuntime::initRuntime(runtime_name);

    switch (node_type) {
    case 'p':
        payload_size = stoul(argv[4]);
        runAsPublisher(id, output_name, payload_size);
        break;
    case 's':
        runAsSubscriber(id, output_name);
        break;
    default:
        cerr << "Unknown type" << endl;
        return 1;
    }

    if (pid == 0) return 0;
    for (int i = 1; i < nproc; i++) wait(nullptr);

    return 0;
}

void runAsPublisher(int id, const char* output_name, size_t payload_size) {
    hlp::Timer timer(env::kTimes);

    try {
        Publisher pub(env::kTopic, env::kPoolSize);

        for (int t = 0; t < env::kTimes; t++) {
            int tag = (id - 1) * env::kTimes + t + 1;
            int* buf_d = (int*) pub.malloc(payload_size);
            init::fillArray(buf_d, payload_size / sizeof(int), tag);

            timer.setPoint(tag);
            // since we won't use the data from the subscriber side, it's ok to
            // exploit the size field to send the tag
            pub.put(buf_d, (size_t) tag);
            // another time point is set at the subscriber-end

            // control publishing frequency
            this_thread::sleep_for(env::kPubInterval);
        }
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }

    stringstream ss;
    ss << output_name << "-" << id << ".csv";
    timer.dump(ss.str().c_str());
}

void runAsSubscriber(int id, const char* output_name) {
    hlp::Timer timer(10000);

    try {
        auto handler = [&timer](void* msg, size_t tag) {
            // upon received, set the current time point
            timer.setPoint((int) tag);
        };
        Subscriber sub(env::kTopic, env::kPoolSize, handler);

        if (id == 1) cout << "Ctrl+C to leave" << endl;
        hlp::waitForSigInt();
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }

    stringstream ss;
    ss << output_name << "-" << id << ".csv";
    timer.dump(ss.str().c_str());
}