#include "profiling.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace profiling {

Records records;

void dump_records(const std::string& name, int points_per_group) {
    std::string output_name = name + ".csv";
    std::ofstream out(output_name);

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

    std::cout << output_name << " is created." << std::endl;
}

}  // namespace profiling