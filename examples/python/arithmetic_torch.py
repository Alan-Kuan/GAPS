import argparse
import signal

import pyshoz

TOPIC = "arithmetic_python"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 4 * 1024 * 1024;  # 4 MiB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        action="store_true",
                        help="be a publisher or not (if not specify, it becomes a subscriber)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda sig, frame: print('Stopped'))

    if args.p:
        print("Publisher Mode")
        run_as_publisher()
    else:
        print("Subscriber Mode")
        run_as_subscriber()

def run_as_publisher():
    publisher = pyshoz.Publisher(TOPIC, LLOCATOR, POOL_SIZE)

    # int[64]
    t = publisher.malloc(1, (64, ), (0, 32, 1))
    for i in range(64):
        t[i] = i
    publisher.put(t)

    # double[16]
    t = publisher.malloc(1, (16, ), (2, 32, 1))
    for i in range(16):
        t[i] = i
    publisher.put(t)

    print("Ctrl+C to leave")
    signal.pause()

def run_as_subscriber():
    subscriber = pyshoz.Subscriber(TOPIC, LLOCATOR, POOL_SIZE)

    def msg_handler(tensor):
        print(tensor * 2)
    subscriber.sub(msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

if __name__ == "__main__":
    main()