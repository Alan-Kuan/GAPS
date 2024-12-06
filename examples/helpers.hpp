#include <cstdint>
#include <ctime>

#ifdef DEBUG
#include <iostream>
#endif

namespace hlp {

class Timer {
public:
    Timer(size_t capacity);
    Timer();
    ~Timer();

    inline void reset() { this->size = 0; }

    inline void setPoint(uint16_t label) {
#ifdef DEBUG
        if (this->size >= capacity) {
            std::cerr << "The timer's buffer overflowed" << std::endl;
            exit(1);
        }
#endif
        clock_gettime(CLOCK_MONOTONIC, this->time_points + size);
        this->labels[size] = label;
        size++;
    }

    // dump all time points to a file
    void dump(const char* filename);

private:
    size_t capacity;
    size_t size;
    struct timespec* time_points;
    uint16_t* labels;
};

// block until Ctrl+C is sent
void waitForSigInt();

}  // namespace hlp
