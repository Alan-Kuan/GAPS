#!/bin/bash

set -e

SCRIPT_DIR=`dirname $(realpath "$0")`
RUN_TEST="./build/examples/latency/run_test"
OUTPUT_DIR="outputs/1p1s"

SIZE=(4096 16384 65536 262144 1048576 4194304)
NAME=(4KB 16KB 64KB 256KB 1MB 4MB)
TIMES=100
PUB_INTERVAL=0.0001  # 100 us

mkdir -p "${OUTPUT_DIR}"

./build/src/mem_manager >/dev/null &
MM_PID="$!"
sleep 1

trap -- "kill -s INT ${MM_PID}" EXIT

echo "Starting GAPS Latency Test"
echo

for i in "${!SIZE[@]}"; do
    echo "Testing with payload size: ${NAME[i]} ..."

    PUB_PREFIX="${OUTPUT_DIR}/pub-${NAME[i]}"
    SUB_PREFIX="${OUTPUT_DIR}/sub-${NAME[i]}"

    "${RUN_TEST}" -n 1 -o "${SUB_PREFIX}" >/dev/null &
    SUB_PID="$!"
    sleep 1  # wait a while for the subscriber to be ready
    "${RUN_TEST}" -n 1 -o "${PUB_PREFIX}" -p -s "${SIZE[i]}" -t "${TIMES}" -i "${PUB_INTERVAL}" >/dev/null
    sleep 1  # wait a while for the subscriber to finish handling

    kill -s INT "${SUB_PID}"
    wait "${SUB_PID}"
done

echo
echo "Done!"

"${SCRIPT_DIR}/format_latency_test_results.py" "${OUTPUT_DIR}"