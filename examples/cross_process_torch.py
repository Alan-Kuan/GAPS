import argparse
import torch

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
    publisher = pyshoz.Publisher("cross_process_torch", LLOCATOR, POOL_SIZE)

    t = torch.tensor(range(64), dtype=torch.int)
    publisher.put(t)

    t = torch.tensor(range(16), dtype=torch.float)
    publisher.put(t)

def run_as_subscriber():
    subscriber = pyshoz.Subscriber("cross_process_torch", LLOCATOR, POOL_SIZE)

    def msg_handler(tensor):
        print(tensor * 2)
    subscriber.sub(msg_handler)
    input("Type enter to continue\n")

if __name__ == "__main__":
    main()