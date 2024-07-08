#include <sys/time.h>
#include <functional>
#include <ctime>
#include <chrono>

using namespace std::chrono;

namespace hlp {

class TimeHelper {
public:
    TimeHelper() : tv({0, 0}) {}

    inline double getMSec() {
        return (double) this->tv.tv_sec * 1000.0 
             + (double) this->tv.tv_usec / 1000.0;
    }

    inline void setPoint() {
        gettimeofday(&this->tv, 0);
    } 
private:
    timeval tv;
};

class StdTimeHelper {
public:

    inline void setPoint() {
        this->timePoint = system_clock::now();
    }

    inline system_clock::time_point getPoint() {
        return this->timePoint;
    }

    inline char* getClock() {
        std::time_t tt system_clock::to_time_t(this->timePoint);
        return std::ctime(&tt);
    }
    
    inline milliseconds interval(StdTimeHelper helper) {
        auto dr = (this->timePoint - helper.getPoint());
        return duration_cast<milliseconds>(dr);
    }

private:
    system_clock::time_point timePoint;
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

}
