#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
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

void runAsPublisher(int id, const char* output_name, size_t payload_size,
                    int times, double pub_interval);
void runAsSubscriber(int id, const char* output_name);

void printUsageAndExit(const char* arg0);

int main(int argc, char* argv[]) {
    bool is_publisher = false;
    int nproc = -1;
    const char* output_name = nullptr;
    size_t payload_size = 0;
    int times = -1;
    double pub_interval = -1;
    int opt;

    while ((opt = getopt(argc, argv, "pn:o:s:t:i:")) != -1) {
        switch (opt) {
        case 'p':
            is_publisher = true;
            break;
        case 'n':
            nproc = stoi(optarg);
            break;
        case 'o':
            output_name = optarg;
            break;
        case 's':
            payload_size = stoul(optarg);
            break;
        case 't':
            times = stoi(optarg);
            break;
        case 'i':
            pub_interval = stod(optarg);
            break;
        default:
            printUsageAndExit(argv[0]);
        }
    }

    if (nproc < 0) {
        cout << "-n should be specified" << endl;
        exit(1);
    }
    if (!output_name) {
        cout << "-o should be specifed" << endl;
        exit(1);
    }
    if (is_publisher) {
        if (payload_size == 0) {
            cout << "-s should be specified" << endl;
            exit(1);
        }
        if (times < 0) {
            cout << "-t should be specified" << endl;
            exit(1);
        }
        if (pub_interval < 0) {
            cout << "-i should be specified" << endl;
            exit(1);
        }
    }

#ifdef DEBUG
    cout << "The code was compiled with DEBUG macro defined, which may "
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
    sprintf(runtime_name, "latency_test_%c", is_publisher ? 'p' : 's');
    iox::runtime::PoshRuntime::initRuntime(runtime_name);

    if (is_publisher) {
        runAsPublisher(id, output_name, payload_size, times, pub_interval);
    } else {
        runAsSubscriber(id, output_name);
    }

    if (pid == 0) return 0;
    for (int i = 1; i < nproc; i++) wait(nullptr);

    return 0;
}

void runAsPublisher(int id, const char* output_name, size_t payload_size,
                    int times, double pub_interval) {
    int total_times = 3 + times;
    hlp::Timer timer(total_times);

    try {
        Publisher pub(env::kTopic, env::kPoolSize, env::kMsgQueueCapExp);

        // warming up
        for (int t = 0; t < 3; t++) {
            int tag = (id - 1) * total_times + t + 1;
            int* buf_d = (int*) pub.malloc(payload_size);
            init::fillArray(buf_d, payload_size / sizeof(int), tag);
            timer.setPoint(tag);
            pub.put(buf_d, (size_t) tag);
            this_thread::sleep_for(1s);
        }

        for (int t = 3; t < total_times; t++) {
            int tag = (id - 1) * total_times + t + 1;
            int* buf_d = (int*) pub.malloc(payload_size);
            init::fillArray(buf_d, payload_size / sizeof(int), tag);

            timer.setPoint(tag);
            // since we won't use the data from the subscriber side, it's ok to
            // exploit the size field to send the tag
            pub.put(buf_d, (size_t) tag);
            // another time point is set at the subscriber-end

            // control publishing frequency
            this_thread::sleep_for(chrono::duration<double>(pub_interval));
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
            timer.setPoint(tag);
        };
        Subscriber sub(env::kTopic, env::kPoolSize, env::kMsgQueueCapExp,
                       handler);

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

void printUsageAndExit(const char* arg0) {
    cerr << "usage: " << arg0 << " [-p] -n N -o O [-s S] [-t T] [-i I]\n\n"
         << "options:\n"
         << "  -p\tbe a publisher or not (if not specify, it becomes a "
            "subscriber)\n"
         << "  -n N\tnumber of publishers / subscribers\n"
         << "  -o O\tprefix of the output csv (may contain directory)\n"
         << "  -s S\tsize of the payload to be published (only required if -p "
            "is "
            "specified)\n"
         << "  -t T\tpublishing how many times (only required if -p is "
            "specified)\n"
         << "  -i I\tpublishing interval in second (only required if -p is "
            "specified)"
         << endl;
    exit(1);
}