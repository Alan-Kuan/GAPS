#ifndef ENV_HPP
#define ENV_HPP

#include <cstddef>

namespace env {

const char kTopic[] = "latency-test";
const char kDftLLocator[] = "udp/224.0.0.123:7447#iface=lo";
constexpr size_t kPoolSize = 32 << 20;  // 32 MiB
const int kMsgQueueCapExp = 7;

}  // namespace env

#endif  // ENV_HPP