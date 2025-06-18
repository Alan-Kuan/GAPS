#!/bin/bash

set -e

SCRIPT_DIR=`dirname $(realpath "$0")`
RUN_TEST="./build/examples/latency/run_test"
OUTPUT_DIR="outputs/mpns"

NP=(1 1 1 2 4 8)
NS=(2 4 8 1 1 1)
SIZE=65536
TIMES=100
PUB_INTERVAL=0.0001  # 100 us

mkdir -p "${OUTPUT_DIR}"

iox-roudi -l off &
ROUDI_PID="$!"

./build/src/mem_manager >/dev/null &
MM_PID="$!"
sleep 1

trap -- "kill -s INT ${ROUDI_PID} ${MM_PID}" EXIT

echo "Starting GAPS Latency MPNS Test"
echo

for i in "${!NP[@]}"; do
    echo "Testing with (p, s) = (${NP[i]}, ${NS[i]}) ..."

    PUB_PREFIX="${OUTPUT_DIR}/pub-${NP[i]}p${NS[i]}s"
    SUB_PREFIX="${OUTPUT_DIR}/sub-${NP[i]}p${NS[i]}s"

    "${RUN_TEST}" -n "${NS[i]}" -o "${SUB_PREFIX}" >/dev/null &
    SUB_PID="$!"
    sleep 1  # wait a while for the subscriber to be ready
    "${RUN_TEST}" -n "${NP[i]}" -o "${PUB_PREFIX}" -p -s "${SIZE}" -t "${TIMES}" -i "${PUB_INTERVAL}" >/dev/null
    sleep 1  # wait a while for the subscriber to finish handling

    kill -s INT "-${SUB_PID}"  # send SIGINT to all subscribers
    wait "${SUB_PID}"
done

echo
echo "Done!"

"${SCRIPT_DIR}/format_latency_mpns_test_results.py" "${OUTPUT_DIR}"