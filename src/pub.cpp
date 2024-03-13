#include "pub.hpp"

#include <cstdlib>
#include <iostream>

#include "zenoh.hxx"

#include "work.hpp"

void runAsPublisher(const char* conf_path) {
    try {
        auto config = zenoh::expect<zenoh::Config>(zenoh::config_from_file(conf_path));
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
    } catch (const zenoh::ErrorMessage& ex) {
        std::cerr << "Zenoh: " << ex.as_string_view() << std::endl;
        exit(1);
    }
}
