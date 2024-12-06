#include "helpers.hpp"

#include <signal.h>

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>

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

    logfile << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < this->size; i++) {
        double timepoint = this->time_points[i].tv_sec * 1000 +
                           this->time_points[i].tv_nsec / 1000.0;
        logfile << this->labels[i] << "," << timepoint << std::endl;
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