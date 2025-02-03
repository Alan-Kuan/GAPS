#ifndef ENV_HPP
#define ENV_HPP

#include <cstddef>

namespace env {

const char kTopicName[] = "video";
constexpr size_t kPoolSize = 4 << 20;  // 4 MiB
const int kMsgQueueCapExp = 7;

}  // namespace env

#endif  // ENV_HPP