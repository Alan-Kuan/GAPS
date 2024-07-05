#include <sys/time.h>
#include <functional>

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
