import os
import signal
import sys
import time

import torch
from transformers import MobileViTForSemanticSegmentation

import pyshoz

DEVICE = "cuda"
TOPIC = "cv_pipeline"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 128 << 20  # 128 MiB
MSG_QUEUE_CAP_EXP = 7

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} EXPECT_IMG_NUM")
        exit(1)

    expect_img_num = int(sys.argv[1])

    signal.signal(signal.SIGINT, lambda sig, frame: print("Stopped"))

    model_name = "apple/deeplabv3-mobilevit-small"
    model = MobileViTForSemanticSegmentation.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    session = pyshoz.ZenohSession(LLOCATOR)

    count = 0
    def msg_handler(inputs):
        nonlocal count

        # inference
        with torch.no_grad():
            logits = model(pixel_values=inputs).logits
            predicted_masks = logits.argmax(1)

        # record end time
        count += inputs.shape[0]
        if count == expect_img_num:
            print(time.monotonic())
            os.kill(os.getpid(), signal.SIGINT)
    _sub = pyshoz.Subscriber(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

if __name__ == "__main__":
    main()