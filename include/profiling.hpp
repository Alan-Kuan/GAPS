#ifndef PROFILING_HPP
#define PROFILING_HPP

#ifdef PROFILING

#include <ctime>
#include <iostream>
#include <string>

#define PROF_WARN                                                    \
    std::cout << "Warn: the library was built with profiling codes!" \
              << std::endl

#define PROF_ADD_POINT             \
    clock_gettime(CLOCK_MONOTONIC, \
                  &profiling::records.tps[profiling::records.tp_idx++])

#define PROF_ADD_TAG(tag) \
    profiling::records.tags[profiling::records.tag_idx++] = tag

namespace profiling {

struct Records {
    struct timespec tps[2000];
    int tp_idx = 0;
    int tags[1000];
    int tag_idx = 0;
};

extern struct Records records;

void dump_records(const std::string& name, int points_per_group);

}  // namespace profiling

#else

#define PROF_WARN
#define PROF_ADD_POINT
#define PROF_ADD_TAG(tag)

#endif  // PROFILING

#endif  // PROFILING_HPP