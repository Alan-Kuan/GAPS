#include "sub.hpp"

#include <cstdlib>
#include <iostream>

#include "zenoh.hxx"

#include "work.hpp"

void runAsSubscriber(const char* conf_path) {
    try {
        auto config = zenoh::expect<zenoh::Config>(zenoh::config_from_file(conf_path));
        auto session = zenoh::expect<zenoh::Session>(zenoh::open(std::move(config)));
        auto subscriber = zenoh::expect<zenoh::Subscriber>(session.declare_subscriber(
            "shoz/cuda", messageHandler
        ));

        std::getchar();
    } catch (const zenoh::ErrorMessage& ex) {
        std::cerr << "Zenoh: " << ex.as_string_view() << std::endl;
        exit(1);
    }
}

void messageHandler(const zenoh::Sample& sample) {
    std::cout << "Received" << std::endl;

    zenoh::BytesView msg = sample.get_payload();
    int* buf = (int*) msg.as_string_view().data();

    copyDataToDeviceAndRun(buf, msg.get_len() / sizeof(int));
}
