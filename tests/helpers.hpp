#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace hlp {

class Timer {
public:
    using clock = std::chrono::system_clock;

    Timer(size_t capacity) : capacity(capacity), size(0) {
        this->time_points = new clock::time_point[capacity];
    }
    Timer() : Timer(64) {}
    ~Timer() { delete[] time_points; }

    inline void reset() { this->size = 0; }

    inline void setPoint() {
        if (this->size >= capacity) {
            std::cerr << "recorder is full" << std::endl;
            exit(1);
        }
        time_points[size++] = clock::now();
    }

    // dump all time points to a file
    void dump(const char* filename) {
        std::ofstream logfile;

        logfile.open(filename);

        logfile << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < this->size; i++) {
            std::chrono::duration<double, std::milli> dur =
                this->time_points[i].time_since_epoch();
            logfile << dur.count() << std::endl;
        }

        logfile.close();
    }

private:
    size_t capacity;
    size_t size;
    clock::time_point* time_points;
};

}  // namespace hlp
