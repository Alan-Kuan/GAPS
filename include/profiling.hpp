#ifndef PROFILING_HPP
#define PROFILING_HPP

#ifdef PROFILING

#include <unistd.h>

#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

#define PROFILE_WARN                                                 \
    std::cout << "Warn: the library was built with profiling codes!" \
              << std::endl
#define PROFILE_INIT(N) struct timespec timepoints[N]
#define PROFILE_SETPOINT(x) clock_gettime(CLOCK_MONOTONIC, timepoints + x)
#define PROFILE_OUTPUT(N, name, tag)                                       \
    do {                                                                   \
        std::stringstream ss;                                              \
        ss << "profile-" << name << '-' << getpid();                       \
        std::ofstream out(ss.str(), std::ios_base::app);                   \
        out << tag << ": ";                                                \
        for (int i = 1; i < N; i++) {                                      \
            double diff =                                                  \
                (timepoints[i].tv_sec - timepoints[i - 1].tv_sec) * 1e6 +  \
                (timepoints[i].tv_nsec - timepoints[i - 1].tv_nsec) / 1e3; \
            out << diff << ' ';                                            \
        }                                                                  \
        out << std::endl;                                                  \
    } while (0)

#else

#define PROFILE_WARN
#define PROFILE_INIT(N)
#define PROFILE_SETPOINT(x)
#define PROFILE_OUTPUT(N, ...)

#endif  // PROFILING

#endif  // PROFILING_HPP