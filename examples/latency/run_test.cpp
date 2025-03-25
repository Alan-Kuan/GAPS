#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <iceoryx_hoofs/log/logmanager.hpp>
#include <iceoryx_posh/runtime/posh_runtime.hpp>

#include "env.hpp"
#include "init.hpp"
#include "node/publisher.hpp"
#include "node/subscriber.hpp"
#include "profiling.hpp"
#include "utils.hpp"

using namespace std;

void runAsPublisher(int id, size_t payload_size, int times,
                    double pub_interval);
void runAsSubscriber(int id);

void printUsageAndExit(const char* arg0);

int main(int argc, char* argv[]) {
#ifndef PROFILING
    cerr << "Please build the project with PROFILING=on to run this program."
         << endl;
    return 1;
#endif

    if (argc == 1) printUsageAndExit(argv[0]);

    bool is_publisher = false;
    int nproc = -1;
    const char* output_prefix = nullptr;
    size_t payload_size = 0;
    int times = -1;
    double pub_interval = -1;
    int opt;

    while ((opt = getopt(argc, argv, "hpn:o:s:t:i:")) != -1) {
        switch (opt) {
        case 'p':
            is_publisher = true;
            break;
        case 'n':
            nproc = stoi(optarg);
            break;
        case 'o':
            output_prefix = optarg;
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
    if (!output_prefix) {
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
    sprintf(runtime_name, "latency_test_%c_%d", is_publisher ? 'p' : 's', id);
    iox::runtime::PoshRuntime::initRuntime(runtime_name);

    if (is_publisher) {
        // It is required by the Zenoh-wrapping version GAPS.
        // We still add this here for controling the same experiment setup.
        this_thread::sleep_for(3s);
        runAsPublisher(id, payload_size, times, pub_interval);
    } else {
        runAsSubscriber(id);
    }

#ifdef PROFILING
    string output_name = string(output_prefix) + '-' + to_string(id);
    int points_per_group = is_publisher ? 4 : 5;
    profiling::dump_records(output_name, points_per_group);
#endif

    if (pid == 0) return 0;
    for (int i = 1; i < nproc; i++) wait(nullptr);

    return 0;
}

void runAsPublisher(int id, size_t payload_size, int times,
                    double pub_interval) {
    int total_times = 3 + times;

    try {
        Publisher pub(env::kTopic, env::kPoolSize, env::kMsgQueueCapExp);

        // warming up
        for (int t = 0; t < 3; t++) {
            int tag = (id - 1) * total_times + t + 1;
            int* buf_d = (int*) pub.malloc(payload_size);
            init::fillArray(buf_d, payload_size / sizeof(int), tag);

            PROF_ADD_TAG(tag);
            PROF_ADD_POINT;
            pub.put(buf_d, tag);

            this_thread::sleep_for(1s);
        }

        // real test
        for (int t = 3; t < total_times; t++) {
            int tag = (id - 1) * total_times + t + 1;
            int* buf_d = (int*) pub.malloc(payload_size);
            init::fillArray(buf_d, payload_size / sizeof(int), tag);

            PROF_ADD_TAG(tag);
            PROF_ADD_POINT;
            // we won't use the payload size at the subscriber side during the
            // test therefore, we exploit the payload_size argument to pass the
            // tag
            pub.put(buf_d, tag);

            // control publishing frequency
            this_thread::sleep_for(chrono::duration<double>(pub_interval));
        }
    } catch (runtime_error& err) {
        cerr << "Publisher: " << err.what() << endl;
        exit(1);
    }
}

void runAsSubscriber(int id) {
    try {
        Subscriber sub(env::kTopic, env::kPoolSize, env::kMsgQueueCapExp,
                       [](void* msg, size_t tag) {
                           // upon received, set the current time point
                           PROF_ADD_POINT;
                           PROF_ADD_TAG(tag);
                       });

        if (id == 1) cout << "Ctrl+C to stop" << endl;
        utils::waitForSigInt();
    } catch (runtime_error& err) {
        cerr << "Subscriber: " << err.what() << endl;
        exit(1);
    }
}

void printUsageAndExit(const char* arg0) {
    cerr << "usage: " << arg0 << " [-h] [-p] -n N -o O [-s S] [-t T] [-i I]\n\n"
         << "options:\n"
         << "  -h\tshow this help message and exit\n"
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