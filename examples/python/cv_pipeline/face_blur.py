import os
import sys
import time

import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
from ultralytics.utils import ops

import pyshoz

DEVICE = "cuda"
TOPIC = "cv_pipeline"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 256 << 20
MSG_QUEUE_CAP_EXP = 7
BLUR_RATIO = 0.25

def main():
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} IMG_DIR MAX_IMG_NUM BATCH_SIZE MODEL_PATH")
        exit(1)

    img_dir = sys.argv[1]
    max_img_num = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    model_path = sys.argv[4]

    model = torch.load(model_path)["model"].to(DEVICE)
    model.eval()

    imgs = [
        torchvision.io.read_image(f"{img_dir}/{p}").to(DEVICE)
        for p in os.listdir(img_dir)[:max_img_num]
    ]
    img_batches = [
        torch.stack(imgs[i:i + batch_size])
        for i in range(0, len(imgs), batch_size)
    ]

    transforms = v2.Compose([
        v2.Resize((640, 640)),
        v2.ToDtype(torch.float16, scale=True),
    ])

    session = pyshoz.ZenohSession(LLOCATOR)
    pub = pyshoz.Publisher(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP)

    beg = time.monotonic()
    for img_batch in img_batches:
        img_batch = transforms(img_batch)
        blur_faces(model, img_batch)

        buf = pub.malloc(img_batch.shape, pyshoz.float16)
        pub.copy_tensor(buf, img_batch.contiguous())
        pub.put(buf)

    print(f"beg: {beg}")

def blur_faces(model, img_batch):
    with torch.no_grad():
        preds = model(img_batch)

    preds = ops.non_max_suppression(preds)
    boxes_batch = [pred[:, :4].cpu().tolist() for pred in preds]

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