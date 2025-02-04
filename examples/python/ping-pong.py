import argparse
import signal
import time

import pyshoz
import torch

TOPIC_PING = "p3-ping"
TOPIC_PONG = "p3-pong"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 2 << 20;  # 2 MiB
MSG_QUEUE_CAP_EXP = 7

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        action="store_true",
                        help="be a publisher or not (if not specify, it becomes a subscriber)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda _sig, _frame: print("Stopped"))

    session = pyshoz.ZenohSession(LLOCATOR)
    if args.p:
        print("Publisher Mode")
        run_as_publisher(session)
    else:
        print("Subscriber Mode")
        run_as_subscriber(session)

def run_as_publisher(session):
    publisher = pyshoz.Publisher(session, TOPIC_PING, POOL_SIZE, MSG_QUEUE_CAP_EXP)

    ori_tensor = publisher.malloc((64, ), pyshoz.int32)
    for i in range(64):
        ori_tensor[i] = i

    def msg_handler(tensor):
        if torch.equal(tensor, ori_tensor * 2):
            print('Passed')
        else:
            print('Failed')

    # NOTE: intentionally assign it to a variable, or it destructs right after this line is executed
    _subscriber = pyshoz.Subscriber(session, TOPIC_PONG, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    # make sure subscriber is ready
    time.sleep(2)

    publisher.put(ori_tensor)

    print("Ctrl+C to leave")
    signal.pause()

def run_as_subscriber(session):
    publisher = pyshoz.Publisher(session, TOPIC_PONG, POOL_SIZE, MSG_QUEUE_CAP_EXP)

    def msg_handler(tensor):
        buf = publisher.malloc((64, ), pyshoz.int32)
        buf.copy_(tensor)
        buf *= 2
        publisher.put(buf)

    # NOTE: intentionally assign it to a variable, or it destructs right after this line is executed
    _subscriber = pyshoz.Subscriber(session, TOPIC_PING, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

if __name__ == "__main__":
    main()