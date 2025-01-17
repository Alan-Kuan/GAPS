import os
import sys
import time

from facenet_pytorch import MTCNN
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.functional as F

import pyshoz

DEVICE = "cuda"
TOPIC = "cv_pipeline"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 128 << 20  # 128 MiB
BLUR_RATIO = 0.25

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} IMG_DIR MAX_IMG_NUM BATCH_SIZE")
        exit(1)

    img_dir = sys.argv[1]
    max_img_num = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    model = MTCNN(
        margin=14,
        factor=0.6,
        keep_all=True,
        device=DEVICE,
    )

    imgs = [
        torchvision.io.read_image(f"{img_dir}/{p}").to(DEVICE)
        for p in os.listdir(img_dir)[:max_img_num]
    ]
    img_batches = [
        torch.stack(imgs[i:i + batch_size])
        for i in range(0, len(imgs), batch_size)
    ]

    transforms = v2.Compose([
        v2.Resize(544),
        v2.CenterCrop(512),
        v2.ToDtype(torch.float32, scale=True),
    ])

    session = pyshoz.ZenohSession(LLOCATOR)
    pub = pyshoz.Publisher(session, TOPIC, POOL_SIZE)

    beg = time.monotonic()
    for img_batch in img_batches:
        blur_faces(model, img_batch)

        # preprocess
        img_batch = transforms(img_batch)

        buf = pub.malloc(img_batch.shape, pyshoz.float32)
        pub.copy_tensor(buf, img_batch.contiguous())
        pub.put(buf)

    print(f"beg: {beg}")

def blur_faces(model, img_batch):
    boxes_batch, _probs = model.detect(img_batch.permute(0, 2, 3, 1))
    if boxes_batch is None:
        return

    # apply gaussian blur to human faces
    for i, boxes in enumerate(boxes_batch):
        for box in boxes:
            x_beg = round(box[1])
            x_end = round(box[3])
            y_beg = round(box[0])
            y_end = round(box[2])
            roi = img_batch[i, :, x_beg:x_end, y_beg:y_end]
            if roi.shape[1] == 0 or roi.shape[2] == 0:
                continue
            kernel_size = 2 * round(BLUR_RATIO * min(x_end - x_beg, y_end - y_beg) / 2) + 1
            img_batch[i, :, x_beg:x_end, y_beg:y_end] = F.gaussian_blur(roi, kernel_size)

if __name__ == "__main__":
    main()