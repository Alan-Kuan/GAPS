#include "profiling.hpp"

#include <fstream>
#include <sstream>

namespace profiling {

Records records;

void dump_profiling_records(const char* name, int id, int points_per_group) {
    std::stringstream ss;
    ss << "profile-" << name << '-' << id;
    std::ofstream out(ss.str());

    int j_beg = 0;
    for (int i = 0; i < records.tag_idx; i++) {
        int j_end = j_beg + points_per_group;

        out << records.tags[i] << ": ";
        for (int j = j_beg + 1; j < j_end; j++) {
            double diff =
                (records.tps[j].tv_sec - records.tps[j - 1].tv_sec) * 1e6 +
                (records.tps[j].tv_nsec - records.tps[j - 1].tv_nsec) / 1e3;
            out << diff << ' ';
        }
        out << std::endl;

        j_beg = j_end;
    }
}

}  // namespace profiling