import argparse

import pyshoz

LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 4 * 1024 * 1024;  # 4 MiB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        action="store_true",
                        help="be a publisher or not (if not specify, it becomes a subscriber)")
    args = parser.parse_args()

    if args.p:
        print("Publisher Mode")
        run_as_publisher()
    else:
        print("Subscriber Mode")
        run_as_subscriber()

def run_as_publisher():
    publisher = pyshoz.Publisher("arithmetic_torch", LLOCATOR, POOL_SIZE)

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

    input('Type enter to continue\n')

def run_as_subscriber():
    subscriber = pyshoz.Subscriber("arithmetic_torch", LLOCATOR, POOL_SIZE)

    def msg_handler(tensor):
        print(tensor * 2)
    subscriber.sub(msg_handler)
    input("Type enter to continue\n")

if __name__ == "__main__":
    main()