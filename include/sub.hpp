#ifndef SUB_HPP
#define SUB_HPP

#include "zenoh.hxx"

void runAsSubscriber(const char* conf_path);
void messageHandler(const zenoh::Sample& sample);

#endif  // SUB_HPP