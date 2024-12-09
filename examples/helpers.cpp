#include "helpers.hpp"

#include <signal.h>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>

hlp::Timer::Timer(size_t capacity) : capacity(capacity), size(0) {
    this->time_points = new struct timespec[capacity];
    this->labels = new uint16_t[capacity];
}

hlp::Timer::Timer() : Timer(64) {}

hlp::Timer::~Timer() {
    delete[] this->time_points;
    delete[] this->labels;
}

// dump all time points to a file
void hlp::Timer::dump(const char* filename) {
    std::ofstream logfile(filename);

    for (size_t i = 0; i < this->size; i++) {
        auto& time_point = this->time_points[i];
        logfile << this->labels[i] << ',' << time_point.tv_sec << ','
                << time_point.tv_nsec << std::endl;
    }
}

void hlp::waitForSigInt() {
    sigset_t set;
    int sig;
    sigemptyset(&set);
    sigaddset(&set, SIGINT);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
    sigwait(&set, &sig);
}