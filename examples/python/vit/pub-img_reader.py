import os
import pathlib
import sys
import time

import cv2
import torch
from torchvision.transforms import v2

import pyshoz

TOPIC = "vit"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 12 * 1024 * 1024  # 12 MiB
DEVICE = "cuda"

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} IMG_DIR MAX_IMG_COUNT")
    exit(1)

img_dir_path = pathlib.Path(sys.argv[1])
max_img_count = int(sys.argv[2])

# Start measuring
beg = time.time()

img_paths = sorted(os.listdir(img_dir_path))
for file in img_paths[:max_img_count]:
    img_path = img_dir_path / file
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = v2.functional.to_image(img).to(DEVICE)

    transforms = v2.Compose([
        v2.Resize(544),
        v2.CenterCrop(512),
        v2.ToDtype(torch.float32, scale=True),
    ])
    proc_tensor = transforms(img_tensor)

    publisher = pyshoz.Publisher(TOPIC, LLOCATOR, POOL_SIZE)
    buf_tensor = publisher.malloc(3, (3, 512, 512), (2, 32, 1), False)
    publisher.copy_tensor(buf_tensor, proc_tensor)
    publisher.put(buf_tensor)
    time.sleep(0.1)

print(f"Beginning time: {beg}")