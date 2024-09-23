import os
import signal
import sys
import time

from transformers import MobileViTForImageClassification
import torch
from torchvision.transforms import v2

import pyshoz

TOPIC = "vit"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 12 * 1024 * 1024  # 12 MiB
DEVICE = "cuda"

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} EXPECT_IMG_NUM")
    exit(1)
expect_img_num = int(sys.argv[1])

signal.signal(signal.SIGINT, lambda sig, frame: print("Stopped"))

model_name = "apple/mobilevit-small"
model = MobileViTForImageClassification.from_pretrained(model_name)
model.to(DEVICE)
model.eval()

subscriber = pyshoz.Subscriber(TOPIC, LLOCATOR, POOL_SIZE)
preds = []
timepoints = []

def handler(inputs):
    with torch.no_grad():
        logits = model(pixel_values=inputs.unsqueeze(0)).logits
    pred = logits.argmax(-1).item()
    preds.append(pred)
    timepoints.append(time.time())
    if len(timepoints) == expect_img_num:
        os.kill(os.getpid(), signal.SIGINT)
subscriber.sub(handler)

print(f"Ready! Expect to receive {expect_img_num} images")
signal.pause()

for pred in preds:
    print(model.config.id2label[pred])

print(f"Finish time: {timepoints[-1]}")