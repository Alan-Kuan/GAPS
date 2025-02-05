import argparse
import signal
import time

import pyshoi

TOPIC = "py_latency_test"
POOL_SIZE = 32 << 20;  # 32 MiB
MSG_QUEUE_CAP_EXP = 7
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
    pyshoi.turn_off_logging()

    if args.p:
        if args.s == None:
            print('-s should be specified')
            exit(1)
        pyshoi.init_runtime("py-latency-pub")
        run_as_publisher(int(args.s), args.o)
    else:
        pyshoi.init_runtime("py-latency-sub")
        run_as_subscriber(args.o)

def run_as_publisher(payload_size, output_name):
    publisher = pyshoi.Publisher(TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP)
    count = payload_size // 4
    time_points = [None] * TIMES

    for i in range(TIMES):
        tensor = publisher.empty((count, ), pyshoi.int32)
        tensor.fill_(i)
        time_points[i] = time.monotonic()
        publisher.put(tensor)
        time.sleep(PUB_INTERVAL)

    dump_time_points(time_points, output_name)

def run_as_subscriber(output_name):
    time_points = []

    def msg_handler(_tensor):
        time_points.append(time.monotonic())

    # NOTE: intentionally assign it to a variable, or it destructs right after this line is executed
    _subscriber = pyshoi.Subscriber(TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

    dump_time_points(time_points, output_name)

def dump_time_points(time_points, output_name):
    with open(output_name, 'w') as f:
        for point in time_points:
            f.write(f'{point}\n')

if __name__ == '__main__':
    main()