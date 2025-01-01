#ifndef ENV_HPP
#define ENV_HPP

#include <chrono>
#include <cstddef>

namespace env {

using namespace std::chrono_literals;

const char kTopic[] = "latency-test";
const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
const int kTimes = 100;
const auto kPubInterval = 20ms;
constexpr size_t kPoolSize = 32 * 1024 * 1024;  // 32 MiB

}  // namespace env

#endif  // ENV_HPP