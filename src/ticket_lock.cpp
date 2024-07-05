#include "ticket_lock.hpp"

#include <atomic>

void TicketLock::lock() {
    int me = std::atomic_ref<int>(this->next).fetch_add(1);
    while (this->owner != me);
}

void TicketLock::unlock() {
    this->owner++;
}