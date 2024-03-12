#include <unistd.h>

#include <cstdlib>
#include <iostream>

#include "pub.hpp"
#include "sub.hpp"

void print_usage_and_exit(char* program_name);

int main(int argc, char* argv[]) {
    if (argc != 2) print_usage_and_exit(argv[0]);

    int opt;
    while ((opt = getopt(argc, argv, "ps")) != -1) {
        switch (opt) {
        case 'p':
            runAsPublisher();
            break;
        case 's':
            runAsSubscriber();
            break;
        default:
            print_usage_and_exit(argv[0]);
        }
    }

    return 0;
}

void print_usage_and_exit(char* program_name) {
    std::cerr << "Usage: " << program_name << " <-p|-s>" << std::endl;
    std::exit(1);
}