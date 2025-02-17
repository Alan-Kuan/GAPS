#ifndef PROFILING_HPP
#define PROFILING_HPP

#ifdef PROFILING

#include <ctime>
#include <iostream>

#define PROFILE_WARN                                                 \
    std::cout << "Warn: the library was built with profiling codes!" \
              << std::endl
#define PROFILE_INIT(N) struct timespec timepoints[N]
#define PROFILE_SETPOINT(x) clock_gettime(CLOCK_MONOTONIC, timepoints + x)
#define PROFILE_OUTPUT(N)                                                  \
    do {                                                                   \
        for (int i = 1; i < N; i++) {                                      \
            double diff =                                                  \
                (timepoints[i].tv_sec - timepoints[i - 1].tv_sec) * 1e6 +  \
                (timepoints[i].tv_nsec - timepoints[i - 1].tv_nsec) / 1e3; \
            std::cout << diff << ' ';                                      \
        }                                                                  \
        std::cout << std::endl;                                            \
    } while (0)

#else

#define PROFILE_WARN
#define PROFILE_INIT(N)
#define PROFILE_SETPOINT(x)
#define PROFILE_OUTPUT(N)

#endif  // PROFILING

#endif  // PROFILING_HPP