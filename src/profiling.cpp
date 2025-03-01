#include "profiling.hpp"

#include <fstream>
#include <string>

namespace profiling {

Records records;

void dump_records(std::string& name, int points_per_group) {
    std::ofstream out(name + ".csv");

    int j_beg = 0;
    for (int i = 0; i < records.tag_idx; i++) {
        int j_end = j_beg + points_per_group;

        out << records.tags[i];
        for (int j = j_beg; j < j_end; j++) {
            out << ',' << records.tps[j].tv_sec << ','
                << records.tps[j].tv_nsec;
        }
        out << std::endl;

        j_beg = j_end;
    }
}

}  // namespace profiling