import os
import sys
import time

import torch
import torchvision
from torchvision.transforms import v2

import pygaps

# 
#  This program mocks a node decoding images from a camera with GPU,
#  and sending the preprocessed results to the subscribers.
# 

DEVICE = "cuda"
TOPIC = "cv_pipeline-preprocessed_frames"
RUNTIME = "cv_pipeline-preproc"
POOL_SIZE = 256 << 20
MSG_QUEUE_CAP_EXP = 7

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} IMG_DIR MAX_IMG_NUM BATCH_SIZE")
        exit(1)

    img_dir = sys.argv[1]
    max_img_num = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    # add 3 more batches for warming up
    total_img_num = max_img_num + 3 * batch_size

    # raw image bytes
    raw_imgs = [
        torchvision.io.read_file(f"{img_dir}/{p}")
        for p in os.listdir(img_dir)[:total_img_num]
    ]

    transforms = v2.Compose([
        v2.Resize((512, 512)),
        v2.ToDtype(torch.float16, scale=True),
    ])

    pygaps.turn_off_logging()
    pygaps.init_runtime(RUNTIME)
    pub = pygaps.Publisher(TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP)

    # warming up
    for i in range(0, 3 * batch_size, batch_size):
        img_batch = torchvision.io.decode_jpeg(raw_imgs[i:i + batch_size], device=DEVICE)
        img_batch = torch.stack(img_batch)
        img_batch = transforms(img_batch)

        buf = pub.empty(img_batch.shape, pygaps.float16)
        buf.copy_(img_batch)
        pub.put(buf)
    time.sleep(1)

    beg = time.monotonic()
    for i in range(3 * batch_size, len(raw_imgs), batch_size):
        img_batch = torchvision.io.decode_jpeg(raw_imgs[i:i + batch_size], device=DEVICE)
        img_batch = torch.stack(img_batch)
        img_batch = transforms(img_batch)

        buf = pub.empty(img_batch.shape, pygaps.float16)
        buf.copy_(img_batch)
        pub.put(buf)

    print(f"beg: {beg}")

if __name__ == "__main__":
    main()