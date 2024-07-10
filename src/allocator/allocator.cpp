#include "allocator/allocator.hpp"

#include "error.hpp"

size_t Allocator::malloc(size_t size) {
    if (!this->allocator) throwError("No allocator");
    return this->allocator->malloc(size);
}

void Allocator::free(size_t offset) {
    if (!this->allocator) throwError("No allocator");
    this->allocator->free(offset);
}