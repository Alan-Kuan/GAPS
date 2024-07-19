#include <sys/time.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

namespace hlp {

class TimePoint {
public:
    TimePoint() : tv({0, 0}) {}
    TimePoint(timeval t) : tv(t) {}

    inline void set() { gettimeofday(&tv, 0); }

    inline double getMSec() {
        return (double) tv.tv_sec * 1000.0 + (double) tv.tv_usec / 1000.0;
    }

    TimePoint operator-(TimePoint const& obj) {
        return TimePoint(
            {tv.tv_sec - obj.tv.tv_sec, tv.tv_usec - obj.tv.tv_usec});
    }

private:
    timeval tv;
};

class Timer {
public:
    Timer() : capacity(64), size(0) {
        this->recorder = new TimePoint[this->capacity];
    }

    Timer(size_t capacity) : size(0) {
        this->capacity = capacity;
        this->recorder = new TimePoint[capacity];
    }

    /* record a time point and store in an array "recorder"
     */
    inline void setPoint() {
        if (this->size >= capacity) {
            std::cerr << "recorder is full\n";
            exit(1);
        }
        this->recorder[this->size++].set();
    }

    /* let the recorder be empty size.
     */
    inline void reset() { this->size = 0; }

    /* return a time point according to index;
     */
    inline double getMSec(size_t tpIndex) {
        return this->recorder[tpIndex].getMSec();
    }

    /* print all the time points store in array
     */
    inline void showAll(const char* prefix) {
        for (size_t i = 0; i < this->size; i++) {
            std::cout << prefix << ": " << std::fixed << this->getMSec(i)
                      << "\n";
        }
    }

    /* write all time point to a file
     */
    inline void writeAll(const char* filename) {
        std::ofstream logfile;
        logfile.open(filename);
        for (size_t i = 0; i < this->size; i++) {
            logfile << std::fixed << std::setprecision(3) << this->getMSec(i)
                    << "\n";
        }
        logfile.close();
    }

    inline size_t getSize() { return this->size; }

    static inline double calcDuration(TimePoint t1, TimePoint t2) {
        return (t2 - t1).getMSec();
    }

    ~Timer() { delete[] this->recorder; }

private:
    // timeval tv;
    size_t size;  // time point counter
    size_t capacity;
    TimePoint* recorder;
};

class StdTimer {
public:
    using clock = std::chrono::steady_clock;
    using sec = std::chrono::seconds;
    using millisec = std::chrono::milliseconds;
    using microsec = std::chrono::microseconds;

    StdTimer(size_t capacity) : timePointCounter(0) {
        recorder = new clock::time_point[capacity];
    }

    inline void setPoint() {
        recorder[timePointCounter] = clock::now();
        timePointCounter++;
    }

    inline clock::time_point getPoint(size_t pointIndex) {
        return recorder[pointIndex];
    }

    template <typename T>
    inline T getDuration(size_t t1Idx, size_t t2Idx) {
        auto t1 = recorder[t1Idx];
        auto t2 = recorder[t2Idx];
        return std::chrono::duration_cast<T>(t1 - t2);
    }

    ~StdTimer() { delete[] recorder; }

    StdTimer() = delete;
    StdTimer(const StdTimer&) = delete;

private:
    size_t timePointCounter;
    clock::time_point* recorder;
};

}  // namespace hlp
