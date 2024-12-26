#include "zenoh_wrapper.hpp"

#include <zenoh.hxx>

const zenoh::Session& ZenohSession::getSession() const { return this->session; }

zenoh::Config ZenohSession::makeConfig(const char* llocator) const {
    auto config = zenoh::Config::create_default();
    config.insert(Z_CONFIG_MODE_KEY, Z_CONFIG_MODE_PEER);
    config.insert(Z_CONFIG_LISTEN_KEY, llocator);
    return config;
}