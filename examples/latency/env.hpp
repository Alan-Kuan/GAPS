#ifndef ENV_HPP
#define ENV_HPP

#include <chrono>
#include <cstddef>

namespace env {

using namespace std::chrono_literals;

const char kTopic[] = "latency-test";
const int kTimes = 100;
const auto kPubInterval = 20ms;
constexpr size_t kPoolSize = 32 << 20;  // 32 MiB
const int kMsgQueueCapExp = 7;

}  // namespace env

#endif  // ENV_HPP