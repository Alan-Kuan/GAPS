#!/bin/bash

set -e

SIZE=(1024 4096 16384 65536 262144 1048576 4194304 16777216)
NAME=(1KB 4KB 16KB 64KB 256KB 1MB 4MB 16MB)

mkdir -p outputs
./build/src/mem_manager >/dev/null &
MM_PID="$!"
sleep 1

trap -- "kill -s INT ${MM_PID}" EXIT

for i in {0..7}; do
    echo "Testing with payload size: ${NAME[i]} ..."

    ./build/examples/latency/run_test 1 1 "outputs/sub-${NAME[i]}.csv" >/dev/null &
    SUB_PID="$!"
    sleep 1  # wait until subscriber is ready
    ./build/examples/latency/run_test 0 1 "outputs/pub-${NAME[i]}.csv" "${SIZE[i]}" 100 >/dev/null

    kill -s INT "${SUB_PID}"
    sleep 1  # wait until subscriber finish dumping
done
echo "Done!"