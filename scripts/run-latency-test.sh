#!/bin/bash

set -e

TIMES=100
SIZE=(1024 4096 16384 65536 262144 1048576 4194304 16777216)
NAME=(1KB 4KB 16KB 64KB 256KB 1MB 4MB 16MB)

NP=(2 4 8 1 1 1 4)
NS=(1 1 1 2 4 8 4)
SPEC_SIZE="${SIZE[7]}"

mkdir -p outputs/1p1s
mkdir -p outputs/mpns

./build/src/mem_manager >/dev/null &
MM_PID="$!"
sleep 1

trap -- "kill -s INT ${MM_PID}" EXIT

echo "1. Testing with different payload sizes"
echo

for i in {0..7}; do
    echo "Testing with payload size: ${NAME[i]} ..."

    ./build/examples/latency/run_test s 1 "outputs/1p1s/sub-${NAME[i]}" >/dev/null &
    SUB_PID="$!"
    sleep 1  # wait a while for the subscriber to be ready
    ./build/examples/latency/run_test p 1 "outputs/1p1s/pub-${NAME[i]}" "${SIZE[i]}" "${TIMES}"
    sleep 1  # wait a while for the subscriber to finish handling

    kill -s INT "${SUB_PID}"
    sleep 1  # wait a while for the subscriber to finish dumping
done

echo
echo "2. Testing with different numbers of publishers and subscribers"
echo

for i in {0..6}; do
    echo "Testing with (p, s) = (${NP[i]}, ${NS[i]}) ..."

    ./build/examples/latency/run_test s "${NS[i]}" "outputs/mpns/sub-${NP[i]}p${NS[i]}s" >/dev/null &
    sleep 1  # wait a while for the subscriber to be ready
    ./build/examples/latency/run_test p "${NP[i]}" "outputs/mpns/pub-${NP[i]}p${NS[i]}s" "${SPEC_SIZE}" "${TIMES}"
    sleep 1  # wait a while for the subscriber to finish handling

    pkill --signal INT run_test
    sleep 1  # wait a while for the subscriber to finish dumping
done

echo
echo "Done!"