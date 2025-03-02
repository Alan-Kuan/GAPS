import argparse
import signal
import time

import pygaps

#
#  This program is used to profile the publishing loop at the Python side
#

TOPIC = "py_latency_test"
POOL_SIZE = 32 << 20;  # 32 MiB
MSG_QUEUE_CAP_EXP = 7

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        action="store_true",
                        help="be a publisher or not (if not specify, it becomes a subscriber)")
    parser.add_argument("-s",
                        help="size of the payload to be published (only required if -p is specified)")
    parser.add_argument("-t",
                        help="publishing how many times (only required if -p is specified)")
    parser.add_argument("-i",
                        help="publishing interval in second (only required if -p is specified)")
    parser.add_argument("-o",
                        required=True,
                        help="prefix of the output csv (may contain directory)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda _sig, _frame: print("Stopped"))
    pygaps.turn_off_logging()

    if args.p:
        if args.s == None:
            print("-s should be specified")
            exit(1)
        if args.t == None:
            print("-t should be specified")
            exit(1)
        if args.i == None:
            print("-i should be specified")
            exit(1)
        pygaps.init_runtime("py-latency-pub")
        run_as_publisher(args.o, int(args.s), int(args.t), float(args.i))
    else:
        pygaps.init_runtime("py-latency-sub")
        run_as_subscriber()

    if hasattr(pygaps, "profiling"):
        pygaps.profiling.dump_records(args.o + "-cpp", 3 if args.p else 4)

def run_as_publisher(output_prefix, payload_size, times, pub_interval):
    publisher = pygaps.Publisher(TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP)
    count = payload_size // 4
    total_times = times + 3
    time_points_1 = [0] * total_times
    time_points_2 = [0] * total_times
    time_points_3 = [0] * total_times

    # warming up
    for i in range(3):
        time_points_1[i] = time.monotonic()
        tensor = publisher.empty((count, ), pygaps.int32)
        tensor.fill_(i)
        time_points_2[i] = time.monotonic()
        publisher.put(tensor)
        time_points_3[i] = time.monotonic()
        time.sleep(1)

    for i in range(3, total_times):
        time_points_1[i] = time.monotonic()
        tensor = publisher.empty((count, ), pygaps.int32)
        tensor.fill_(i)
        time_points_2[i] = time.monotonic()
        publisher.put(tensor)
        time_points_3[i] = time.monotonic()
        time.sleep(pub_interval)

    output_name = output_prefix + "-py.csv"
    with open(output_name, "w") as f:
        for i in range(total_times):
            dur_init = (time_points_2[i] - time_points_1[i]) * 1e6
            dur_put = (time_points_3[i] - time_points_2[i]) * 1e6
            f.write(f"{dur_init},{dur_put}\n")
    print(f"{output_name} is created.")

def run_as_subscriber():
    time_points = [0] * 1000
    idx = 0

    def msg_handler(_tensor):
        nonlocal idx
        time_points[idx] = time.monotonic()
        idx += 1

    # NOTE: intentionally assign it to a variable, or it destructs right after this line is executed
    _subscriber = pygaps.Subscriber(TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    print("Ctrl+C to stop")
    signal.pause()

if __name__ == "__main__":
    main()