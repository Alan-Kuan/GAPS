#include "pub.hpp"

#include <iostream>

#include "zenoh.hxx"

#include "work.hpp"

void runAsPublisher(void) {
    zenoh::Config config;
    auto session = zenoh::expect<zenoh::Session>(zenoh::open(std::move(config)));
    auto publisher = zenoh::expect<zenoh::Publisher>(session.declare_publisher("shoz/cuda"));

    int buf[64];
    initAndCopyDataToHost(buf, 64);

    zenoh::BytesView msg((void*) buf, sizeof(buf));

    if (publisher.put(msg)) {
        std::cout << "Published" << std::endl;
    } else {
        std::cout << "Failed to publish" << std::endl;
    }
}