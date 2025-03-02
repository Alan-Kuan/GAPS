#!/bin/bash

set -e

SCRIPT_DIR=`dirname $(realpath "$0")`
PROJECT_DIR=`dirname "${SCRIPT_DIR}"`
TEST="${PROJECT_DIR}/examples/python/latency/run_test_breakdown.py"
CONVERT="${SCRIPT_DIR}/convert.py"
OUTPUT_DIR="outputs/python/breakdown"

SIZE=(1024 4194304)
NAME=(1KB 4MB)
TIMES=10
PUB_INTERVAL=0.0001  # 100 us

mkdir -p "${OUTPUT_DIR}"

./build/src/mem_manager >/dev/null &
MM_PID="$!"
sleep 1

trap -- "kill -s INT ${MM_PID}" EXIT

echo "Starting PyGAPS Latency Breakdown Test"

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
    sleep 1  # wait a while for the subscriber to finish dumping

    # convert profiling records
    "${CONVERT}" "${PUB_PREFIX}-cpp.csv"
    "${CONVERT}" "${SUB_PREFIX}-cpp.csv"
done

echo
echo "Done!"