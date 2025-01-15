import argparse
import signal
import time

import pyshoz

TOPIC = "py_latency_test"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 32 * 1024 * 1024;  # 32 MiB
PUB_INTERVAL = 0.5  # 500ms
TIMES = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        action="store_true",
                        help="be a publisher or not (if not specify, it becomes a subscriber)")
    parser.add_argument("-s",
                        help="size of the payload to be published (only required if -p is specified)")
    parser.add_argument("-o",
                        required=True,
                        help="output path")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda _sig, _frame: print("Stopped"))

    session = pyshoz.ZenohSession(LLOCATOR)
    if args.p:
        if args.s == None:
            print('-s should be specified')
            exit(1)
        run_as_publisher(session, int(args.s), args.o)
    else:
        run_as_subscriber(session, args.o)

def run_as_publisher(session, payload_size, output_name):
    publisher = pyshoz.Publisher(session, TOPIC, POOL_SIZE)
    count = payload_size // 4
    time_points = [None] * TIMES

    for i in range(TIMES):
        tensor = publisher.malloc((count, ), (0, 32, 1))
        tensor.fill_(i)
        time_points[i] = time.monotonic()
        publisher.put(tensor)
        time.sleep(PUB_INTERVAL)

    dump_time_points(time_points, output_name)

def run_as_subscriber(session, output_name):
    time_points = []

    def msg_handler(_tensor):
        time_points.append(time.monotonic())

    # NOTE: intentionally assign it to a variable, or it destructs right after this line is executed
    _subscriber = pyshoz.Subscriber(session, TOPIC, POOL_SIZE, msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

    dump_time_points(time_points, output_name)

def dump_time_points(time_points, output_name):
    with open(output_name, 'w') as f:
        for point in time_points:
            f.write(f'{point}\n')

if __name__ == '__main__':
    main()