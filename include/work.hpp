#ifndef WORK_HPP
#define WORK_HPP

void initAndCopyDataToHost(int* buf, int count);
void copyDataToDeviceAndRun(int* buf, int count);

#endif  // WORK_HPP