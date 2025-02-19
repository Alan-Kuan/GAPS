#!/bin/bash

set -e

SIZE=(1024 4096 16384 65536 262144 1048576 4194304)
NAME=(1KB 4KB 16KB 64KB 256KB 1MB 4MB)

SCRIPT_DIR=`dirname $(realpath "$0")`
PROJECT_DIR=`dirname "${SCRIPT_DIR}"`
TEST="${PROJECT_DIR}/examples/python/latency.py"
OUTPUT_DIR="outputs/python"
TIMES=100
PUB_INTERVAL=0.02  # 20 ms

mkdir -p "${OUTPUT_DIR}"

iox-roudi -l off &
ROUDI_PID="$!"

./build/src/mem_manager >/dev/null &
MM_PID="$!"
sleep 1

trap -- "kill -s INT ${ROUDI_PID} ${MM_PID}" EXIT

echo "1. Testing with different payload sizes"
echo

for i in {0..6}; do
    echo "Testing with payload size: ${NAME[i]} ..."

    python3 "${TEST}" -o "${OUTPUT_DIR}/sub-${NAME[i]}" >/dev/null &
    SUB_PID="$!"
    sleep 1  # wait a while for the subscriber to be ready
    python3 "${TEST}" -o "${OUTPUT_DIR}/pub-${NAME[i]}" -p -s "${SIZE[i]}" -t "${TIMES}" -i "${PUB_INTERVAL}"
    sleep 1  # wait a while for the subscriber to finish handling

    kill -s INT "${SUB_PID}"
    sleep 1  # wait a while for the subscriber to finish dumping
done

echo
echo "Done!"

"${SCRIPT_DIR}/get_python_results.py"