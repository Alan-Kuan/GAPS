#include "sub.hpp"

#include <iostream>

#include "zenoh.hxx"

#include "work.hpp"

void runAsSubscriber(void) {
    zenoh::Config config;
    auto session = zenoh::expect<zenoh::Session>(zenoh::open(std::move(config)));
    auto subscriber = zenoh::expect<zenoh::Subscriber>(session.declare_subscriber(
        "shoz/cuda", messageHandler
    ));

    std::getchar();
}

void messageHandler(const zenoh::Sample& sample) {
    std::cout << "Received" << std::endl;
    zenoh::BytesView msg = sample.get_payload();
    int* buf = (int*) msg.as_string_view().data();

    copyDataToDeviceAndRun(buf, msg.get_len() / sizeof(int));
}