#include <sys/time.h>

#include <chrono>
#include <ctime>
#include <functional>

namespace hlp {

class TimevalPoint {
public:
    TimevalPoint() : tv({0, 0}) {}

    inline double getMSec() {
        return (double) this->tv.tv_sec * 1000.0 
             + (double) this->tv.tv_usec / 1000.0;
    }

    inline void setPoint() {
        gettimeofday(&tv, 0);
    } 
private:
    timeval tv;
};

class TimeHelper {
public:
    TimeHelper(size_t capacity) : tpCounter(0) {
        this->capacity = capacity;
        recorder = new TimevalPoint[capacity];

        for (size_t i = 0; i < capacity; i++) {
            recorder[i] = TimevalPoint();
        }
    } 

    inline void setPoint() {
        recorder[tpCounter % capacity].setPoint();
        tpCounter++;
    }

    inline double getMSec(size_t tpIndex) {
        TimevalPoint tvp = recorder[tpIndex % capacity];
        return tvp.getMSec();
    }

    ~TimeHelper() {
        delete[] this->recorder;
    }
    
    TimeHelper() = delete;
    TimeHelper(const TimeHelper &) = delete;

private:
    // timeval tv;
    size_t tpCounter; // time point counter
    size_t capacity;
    TimevalPoint *recorder;
};

class StdTimeHelper {
public:
    using clock = std::chrono::steady_clock;
    using sec = std::chrono::seconds;
    using millisec = std::chrono::milliseconds;
    using microsec = std::chrono::microseconds;

    StdTimeHelper(size_t capacity) : timePointCounter(0) {
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

    ~StdTimeHelper() {
        delete[] recorder;
    }

    StdTimeHelper() = delete;
    StdTimeHelper(const StdTimeHelper &) = delete;

private:
    size_t timePointCounter;
    clock::time_point *recorder;
};

// static inline void tryCatcher(std::function<void(int)>& f) {
//     try {
//         f();
//     } catch (zenoh::ErrorMessage& err) {
//         cerr << "Zenoh: " << err.as_string_view() << endl;
//         exit(1);
//     } catch (runtime_error& err) {
//         cerr << "Publisher: " << err.what() << endl;
//         exit(1);
//     }
// }

}  // namespace hlp
