import argparse
import signal
import time

import pygaps
import torch

TOPIC_PING = "p3-ping"
TOPIC_PONG = "p3-pong"
POOL_SIZE = 2 << 20;  # 2 MiB
MSG_QUEUE_CAP_EXP = 7

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        action="store_true",
                        help="be a publisher or not (if not specify, it becomes a subscriber)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda _sig, _frame: print("Stopped"))
    pygaps.turn_off_logging()

    if args.p:
        print("Ping Side")
        pygaps.init_runtime("py-ping-pong-publisher")
        run_as_ping_side()
    else:
        print("Pong Side")
        pygaps.init_runtime("py-ping-pong-subscriber")
        run_as_pong_side()

def run_as_ping_side():
    publisher = pygaps.Publisher(TOPIC_PING, POOL_SIZE, MSG_QUEUE_CAP_EXP)

    ori_tensor = publisher.empty((64, ), pygaps.int32)
    for i in range(64):
        ori_tensor[i] = i

    def msg_handler(tensor):
        if torch.equal(tensor, ori_tensor * 2):
            print("Passed")
        else:
            print("Failed")

    # NOTE: intentionally assign it to a variable, or it destructs right after this line is executed
    _subscriber = pygaps.Subscriber(TOPIC_PONG, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    # make sure the subscriber is ready
    time.sleep(2)

    publisher.put(ori_tensor)
    print("Ping!")

    print("Ctrl+C to leave")
    signal.pause()

def run_as_pong_side():
    publisher = pygaps.Publisher(TOPIC_PONG, POOL_SIZE, MSG_QUEUE_CAP_EXP)

    def msg_handler(tensor):
        buf = publisher.empty((64, ), pygaps.int32)
        buf.copy_(tensor)
        buf *= 2
        publisher.put(buf)
        print("Pong!")

    # NOTE: intentionally assign it to a variable, or it destructs right after this line is executed
    _subscriber = pygaps.Subscriber(TOPIC_PING, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

if __name__ == "__main__":
    main()