#include "utils.hpp"

#include <signal.h>

namespace utils {

void waitForSigInt() {
    sigset_t set;
    int sig;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
    sigwait(&set, &sig);
}

}  // namespace utils