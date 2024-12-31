#!/bin/bash

set -e

TIMES=100
SIZE=(1024 4096 16384 65536 262144 1048576 4194304 16777216)
NAME=(1KB 4KB 16KB 64KB 256KB 1MB 4MB 16MB)

NP=(2 4 8 1 1 1 4)
NS=(1 1 1 2 4 8 4)
SPEC_SIZE="${SIZE[7]}"

SCRIPT_DIR=`dirname $(realpath -s "$0")`
PROJECT_DIR=`dirname ${SCRIPT_DIR}`
RUN_TEST="${PROJECT_DIR}/build/examples/latency/run_test"
OUTPUT_1_DIR="${PROJECT_DIR}/outputs/1p1s"
OUTPUT_2_DIR="${PROJECT_DIR}/outputs/mpns"

mkdir -p "${OUTPUT_1_DIR}"
mkdir -p "${OUTPUT_2_DIR}"

iox-roudi -l off -c "${SCRIPT_DIR}/iox-roudi-config.toml" &
ROUDI_PID="$!"

"${PROJECT_DIR}/build/src/mem_manager" >/dev/null &
MM_PID="$!"
sleep 1

trap -- "kill -s INT ${ROUDI_PID}" EXIT
trap -- "kill -s INT ${MM_PID}" EXIT

echo "1. Testing with different payload sizes"
echo

for i in {0..7}; do
    echo "Testing with payload size: ${NAME[i]} ..."

    "${RUN_TEST}" s 1 "${OUTPUT_1_DIR}/sub-${NAME[i]}" >/dev/null &
    SUB_PID="$!"
    sleep 1  # wait a while for the subscriber to be ready
    "${RUN_TEST}" p 1 "${OUTPUT_1_DIR}/pub-${NAME[i]}" "${SIZE[i]}" "${TIMES}"
    sleep 1  # wait a while for the subscriber to finish handling

    kill -s INT "${SUB_PID}"
    sleep 1  # wait a while for the subscriber to finish dumping
done

echo
echo "2. Testing with different numbers of publishers and subscribers"
echo

for i in {0..6}; do
    echo "Testing with (p, s) = (${NP[i]}, ${NS[i]}) ..."

    "${RUN_TEST}" s "${NS[i]}" "${OUTPUT_2_DIR}/sub-${NP[i]}p${NS[i]}s" >/dev/null &
    sleep 1  # wait a while for the subscriber to be ready
    "${RUN_TEST}" p "${NP[i]}" "${OUTPUT_2_DIR}/pub-${NP[i]}p${NS[i]}s" "${SPEC_SIZE}" "${TIMES}"
    sleep 1  # wait a while for the subscriber to finish handling

    pkill --signal INT run_test
    sleep 1  # wait a while for the subscriber to finish dumping
done

echo
echo "Done!"
