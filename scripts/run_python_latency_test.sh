#!/bin/bash

set -e

SCRIPT_DIR=`dirname $(realpath "$0")`
PROJECT_DIR=`dirname "${SCRIPT_DIR}"`
TEST="${PROJECT_DIR}/examples/python/latency/run_test.py"
OUTPUT_DIR="outputs/python/latency"

SIZE=(1024 4096 16384 65536 262144 1048576 4194304)
NAME=(1KB 4KB 16KB 64KB 256KB 1MB 4MB)
TIMES=100
PUB_INTERVAL=0.0001  # 100 us

mkdir -p "${OUTPUT_DIR}"

./build/src/mem_manager >/dev/null &
MM_PID="$!"
sleep 1

trap -- "kill -s INT ${MM_PID}" EXIT

echo "Starting PyGAPS Latency Test"
echo

for i in "${!SIZE[@]}"; do
    echo "Testing with payload size: ${NAME[i]} ..."

    PUB_PREFIX="${OUTPUT_DIR}/pub-${NAME[i]}"
    SUB_PREFIX="${OUTPUT_DIR}/sub-${NAME[i]}"

    python3 "${TEST}" -o "${SUB_PREFIX}" >/dev/null &
    SUB_PID="$!"
    sleep 1  # wait a while for the subscriber to be ready
    python3 "${TEST}" -o "${PUB_PREFIX}" -p -s "${SIZE[i]}" -t "${TIMES}" -i "${PUB_INTERVAL}" >/dev/null
    sleep 1  # wait a while for the subscriber to finish handling

    kill -s INT "${SUB_PID}"
    wait "${SUB_PID}"
done

echo
echo "Done!"

"${SCRIPT_DIR}/format_python_latency_test_results.py" "${OUTPUT_DIR}"