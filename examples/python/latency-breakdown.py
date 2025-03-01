import argparse
import signal
import time

import pygaps

#
#  This program is used to profile the publishing loop at the Python side
#

TOPIC = "py_latency_test"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
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
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda _sig, _frame: print("Stopped"))

    session = pygaps.ZenohSession(LLOCATOR)
    if args.p:
        if args.s == None:
            print('-s should be specified')
            exit(1)
        if args.t == None:
            print("-t should be specified")
            exit(1)
        if args.i == None:
            print('-i should be specified')
            exit(1)

        run_as_publisher(session, int(args.s), int(args.t), float(args.i))
        if hasattr(pygaps, "profiling"):
            pygaps.profiling.dump_records("pub", 1, 3)
    else:
        run_as_subscriber(session)
        if hasattr(pygaps, "profiling"):
            pygaps.profiling.dump_records("sub", 1, 4)

def run_as_publisher(session, payload_size, times, pub_interval):
    publisher = pygaps.Publisher(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP)
    count = payload_size // 4
    total_times = times + 3
    time_points_1 = [None] * total_times
    time_points_2 = [None] * total_times
    time_points_3 = [None] * total_times

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

    print("init put")
    for i in range(total_times):
        dur_init = (time_points_2[i] - time_points_1[i]) * 1e6
        dur_put = (time_points_3[i] - time_points_2[i]) * 1e6
        print(dur_init, dur_put)

def run_as_subscriber(session):
    time_points = []

    def msg_handler(_tensor):
        time_points.append(time.monotonic())

    # NOTE: intentionally assign it to a variable, or it destructs right after this line is executed
    _subscriber = pygaps.Subscriber(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

if __name__ == '__main__':
    main()