#ifndef SUB_HPP
#define SUB_HPP

#include "zenoh.hxx"

void runAsSubscriber(void);
void messageHandler(const zenoh::Sample& sample);

#endif  // SUB_HPP