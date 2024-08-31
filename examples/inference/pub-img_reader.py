import sys

import cv2
import torch
from torchvision.transforms import v2

import pyshoz

LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 8 * 1024 * 1024;  # 8 MiB

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image path>")
        exit(1)

    publisher = pyshoz.Publisher("cross_process_torch", LLOCATOR, POOL_SIZE)

    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transforms = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(img.shape[0]),
        v2.Resize((32, 32)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor = transforms(img)

    publisher.put(tensor)

if __name__ == "__main__":
    main()