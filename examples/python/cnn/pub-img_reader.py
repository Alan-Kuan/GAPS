import sys

import cv2
import torch
from torchvision.transforms import v2

import pyshoz

TOPIC = "cnn"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 8 * 1024 * 1024;  # 8 MiB

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image path>")
        exit(1)

    publisher = pyshoz.Publisher(TOPIC, LLOCATOR, POOL_SIZE)

    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = v2.functional.to_image(img)
    transforms = v2.Compose([
        v2.CenterCrop(img.shape[0]),
        v2.Resize((32, 32)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    proc_tensor = transforms(img_tensor.to('cuda'))

    buf_tensor = publisher.malloc(3, (3, 32, 32), (2, 32, 1), False)
    publisher.copy_tensor(buf_tensor, proc_tensor)

    publisher.put(buf_tensor)

if __name__ == "__main__":
    main()