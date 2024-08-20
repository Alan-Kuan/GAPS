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
        this->labels = new uint16_t[capacity];
    }
    Timer() : Timer(64) {}
    ~Timer() {
        delete[] this->time_points;
        delete[] this->labels;
    }

    inline void reset() { this->size = 0; }

    inline void setPoint(uint16_t label = 0) {
        if (this->size >= capacity) {
            std::cerr << "timer's capacity overflow" << std::endl;
            exit(1);
        }
        this->time_points[size] = clock::now();
        this->labels[size] = label;
        size++;
    }

    // dump all time points to a file
    void dump(const char* filename) {
        std::ofstream logfile;

        logfile.open(filename);

        logfile << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < this->size; i++) {
            std::chrono::duration<double, std::milli> dur =
                this->time_points[i].time_since_epoch();
            logfile << this->labels[i] << "," << dur.count() << std::endl;
        }

        logfile.close();
    }

private:
    size_t capacity;
    size_t size;
    clock::time_point* time_points;
    uint16_t* labels;
};

}  // namespace hlp
