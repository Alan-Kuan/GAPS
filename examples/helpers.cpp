#include "helpers.hpp"

#include <signal.h>

void hlp::waitForSigInt() {
    sigset_t set;
    int sig;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
    sigwait(&set, &sig);
}