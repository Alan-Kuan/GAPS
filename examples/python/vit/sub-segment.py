import os
import signal
import sys
import time

from transformers import MobileViTForSemanticSegmentation
import torch
from torchvision.transforms import v2

import pyshoz

TOPIC = "vit"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 12 << 20  # 12 MiB
MSG_QUEUE_CAP_EXP = 7
DEVICE = "cuda"

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} EXPECT_IMG_NUM")
        exit(1)
    expect_img_num = int(sys.argv[1])

    signal.signal(signal.SIGINT, lambda _sig, _frame: print("Stopped"))

    model_name = "apple/deeplabv3-mobilevit-small"
    model = MobileViTForSemanticSegmentation.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    preds = []
    timepoints = []

    def handler(inputs):
        with torch.no_grad():
            logits = model(pixel_values=inputs.unsqueeze(0)).logits
        predicted_mask = logits.argmax(1).squeeze(0)
        preds.append(predicted_mask)
        timepoints.append(time.time())
        if len(timepoints) == expect_img_num:
            os.kill(os.getpid(), signal.SIGINT)

    session = pyshoz.ZenohSession(LLOCATOR)
    _subscriber = pyshoz.Subscriber(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP, handler)

    print(f"Ready! Expect to receive {expect_img_num} images")
    signal.pause()

    print(f"Finish time: {timepoints[-1]}")

if __name__ == '__main__':
    main()