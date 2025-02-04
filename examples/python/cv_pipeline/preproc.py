import os
import sys
import time

import torch
import torchvision
from torchvision.transforms import v2

import pyshoz

# 
#  This program mocks a node decoding images from a camera with GPU,
#  and sending the preprocessed results to the subscribers.
# 

DEVICE = "cuda"
TOPIC = "cv_pipeline-preprocessed_frames"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 256 << 20
MSG_QUEUE_CAP_EXP = 7
BLUR_RATIO = 0.25

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} IMG_DIR MAX_IMG_NUM BATCH_SIZE")
        exit(1)

    img_dir = sys.argv[1]
    max_img_num = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    # raw image bytes
    raw_imgs = [
        torchvision.io.read_file(f"{img_dir}/{p}")
        for p in os.listdir(img_dir)[:max_img_num]
    ]

    transforms = v2.Compose([
        v2.Resize((640, 640)),
        v2.ToDtype(torch.float16, scale=True),
    ])

    session = pyshoz.ZenohSession(LLOCATOR)
    pub = pyshoz.Publisher(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP)

    beg = time.monotonic()
    for i in range(0, len(raw_imgs), batch_size):
        img_batch = torchvision.io.decode_jpeg(raw_imgs[i:i + batch_size], device=DEVICE)
        img_batch = torch.stack(img_batch)
        img_batch = transforms(img_batch)

        buf = pub.malloc(img_batch.shape, pyshoz.float16)
        pub.copy_tensor(buf, img_batch.contiguous())
        pub.put(buf)

    print(f"beg: {beg}")

if __name__ == "__main__":
    main()