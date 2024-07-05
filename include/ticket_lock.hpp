#ifndef TICKET_LOCK_HPP
#define TICKET_LOCK_HPP

class TicketLock {
public:
    void lock();
    void unlock();

private:
    int next = 0;
    int owner = 0;
};

#endif  // TICKET_LOCK_HPP