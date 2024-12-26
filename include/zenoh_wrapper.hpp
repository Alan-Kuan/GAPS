#ifndef ZENOH_WRAPPER_HPP
#define ZENOH_WRAPPER_HPP

#include <zenoh.hxx>

// a simple zenoh session wrapper for Pyshoz
class ZenohSession {
public:
    ZenohSession() = delete;
    ZenohSession(const char* llocator) : session(this->makeConfig(llocator)) {}
    ZenohSession(const ZenohSession&) = delete;

    const zenoh::Session& getSession() const;

private:
    zenoh::Config makeConfig(const char* llocator) const;

    zenoh::Session session;
};

#endif  // ZENOH_WRAPPER_HPP