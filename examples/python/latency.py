import argparse
import signal
import time

import pygaps

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
    parser.add_argument("-o",
                        required=True,
                        help="output path")
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

        run_as_publisher(session, args.o, int(args.s), int(args.t), float(args.i))
        if hasattr(pygaps, "profiling"):
            pygaps.profiling.dump_records("pub", 1, 3)
    else:
        run_as_subscriber(session, args.o)
        if hasattr(pygaps, "profiling"):
            pygaps.profiling.dump_records("sub", 1, 4)

def run_as_publisher(session, output_name, payload_size, times, pub_interval):
    publisher = pygaps.Publisher(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP)
    count = payload_size // 4
    total_times = times + 3
    time_points = [None] * total_times

    # warming up
    for i in range(3):
        tensor = publisher.empty((count, ), pygaps.int32)
        tensor.fill_(i)
        time_points[i] = time.monotonic()
        publisher.put(tensor)
        time.sleep(1)

    for i in range(3, total_times):
        tensor = publisher.empty((count, ), pygaps.int32)
        tensor.fill_(i)
        time_points[i] = time.monotonic()
        publisher.put(tensor)
        time.sleep(pub_interval)

    dump_time_points(time_points, output_name)

def run_as_subscriber(session, output_name):
    time_points = []

    def msg_handler(_tensor):
        time_points.append(time.monotonic())

    # NOTE: intentionally assign it to a variable, or it destructs right after this line is executed
    _subscriber = pygaps.Subscriber(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

    dump_time_points(time_points, output_name)

def dump_time_points(time_points, output_name):
    with open(output_name, 'w') as f:
        for point in time_points:
            f.write(f'{point}\n')

if __name__ == '__main__':
    main()