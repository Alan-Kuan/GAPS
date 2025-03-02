#!/bin/bash

set -e

SCRIPT_DIR=`dirname $(realpath "$0")`
RUN_TEST="./build/examples/latency/run_test"
OUTPUT_DIR="outputs"
OUTPUT_DIR_1P1S="${OUTPUT_DIR}/1p1s"
OUTPUT_DIR_MPNS="${OUTPUT_DIR}/mpns"

SIZE=(1024 4096 16384 65536 262144 1048576 4194304)
NAME=(1KB 4KB 16KB 64KB 256KB 1MB 4MB)
TIMES=10
PUB_INTERVAL=0.0001  # 100 us

NP=(1 1 2 4 2)
NS=(2 4 1 1 2)
SPEC_SIZE="${SIZE[0]}"

mkdir -p "${OUTPUT_DIR_1P1S}"
mkdir -p "${OUTPUT_DIR_MPNS}"

./build/src/mem_manager >/dev/null &
MM_PID="$!"
sleep 1

trap -- "kill -s INT ${MM_PID}" EXIT

echo "Starting GAPS Latency Test"
echo

echo "1. Testing with different payload sizes"
echo

for i in "${!SIZE[@]}"; do
    echo "Testing with payload size: ${NAME[i]} ..."

    PUB_PREFIX="${OUTPUT_DIR_1P1S}/pub-${NAME[i]}"
    SUB_PREFIX="${OUTPUT_DIR_1P1S}/sub-${NAME[i]}"

    "${RUN_TEST}" -n 1 -o "${SUB_PREFIX}" >/dev/null &
    SUB_PID="$!"
    sleep 1  # wait a while for the subscriber to be ready
    "${RUN_TEST}" -n 1 -o "${PUB_PREFIX}" -p -s "${SIZE[i]}" -t "${TIMES}" -i "${PUB_INTERVAL}" >/dev/null
    sleep 1  # wait a while for the subscriber to finish handling

    kill -s INT "${SUB_PID}"
    sleep 1  # wait a while for the subscriber to finish dumping
done

echo
echo "2. Testing with different numbers of publishers and subscribers"
echo

for i in "${!NP[@]}"; do
    echo "Testing with (p, s) = (${NP[i]}, ${NS[i]}) ..."

    PUB_PREFIX="${OUTPUT_DIR_MPNS}/pub-${NP[i]}p${NS[i]}s"
    SUB_PREFIX="${OUTPUT_DIR_MPNS}/sub-${NP[i]}p${NS[i]}s"

    "${RUN_TEST}" -n "${NS[i]}" -o "${SUB_PREFIX}" >/dev/null &
    sleep 1  # wait a while for the subscriber to be ready
    "${RUN_TEST}" -n "${NP[i]}" -o "${PUB_PREFIX}" -p -s "${SPEC_SIZE}" -t "${TIMES}" -i "${PUB_INTERVAL}" >/dev/null
    sleep 1  # wait a while for the subscriber to finish handling

    pkill --signal INT run_test
    sleep 1  # wait a while for the subscriber to finish dumping
done

echo
echo "Done!"

"${SCRIPT_DIR}/get_results.py" "${OUTPUT_DIR}"