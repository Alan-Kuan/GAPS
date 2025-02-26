import signal
import sys

import torch
import torchvision.transforms.v2.functional as F
from ultralytics.utils import ops

import pygaps

#
#  This is a node that detects and blurs human faces
#

DEVICE = "cuda"
TOPIC_IN = "cv_pipeline-preprocessed_frames"
TOPIC_OUT = "cv_pipeline-blurred_frames"
RUNTIME = "cv_pipeline-blur"
POOL_SIZE = 256 << 20
MSG_QUEUE_CAP_EXP = 7
BLUR_RATIO = 0.25

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} MODEL_PATH")
        exit(1)

    model_path = sys.argv[1]

    signal.signal(signal.SIGINT, lambda _sig, _frame: print("Stopped"))

    model = torch.load(model_path)["model"].to(DEVICE)
    model.eval()

    pygaps.turn_off_logging()
    pygaps.init_runtime(RUNTIME)
    pub = pygaps.Publisher(TOPIC_OUT, POOL_SIZE, MSG_QUEUE_CAP_EXP)

    def msg_handler(img_batch):
        buf = pub.empty(img_batch.shape, pygaps.float16)
        buf.copy_(img_batch)
        blur_faces(model, buf)
        pub.put(buf)
    _sub = pygaps.Subscriber(TOPIC_IN, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

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