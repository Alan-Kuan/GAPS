import signal
import sys
import time

import torch
from ultralytics.utils import ops

import pygaps

#
#  This is a node that does instance segmentation
#

DEVICE = "cuda"
TOPIC = "cv_pipeline-blurred_frames"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 256 << 20
MSG_QUEUE_CAP_EXP = 7

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} MODEL_PATH MAX_IMG_NUM BATCH_SIZE")
        exit(1)

    model_path = sys.argv[1]
    max_img_num = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    expect_img_num = max_img_num + 3 * batch_size

    signal.signal(signal.SIGINT, lambda sig, frame: print("Stopped"))

    model = torch.load(model_path)["model"].to(DEVICE)
    model.eval()

    session = pygaps.ZenohSession(LLOCATOR)

    count = 0
    def msg_handler(inputs):
        nonlocal count

        with torch.no_grad():
            preds = model(inputs)

        input_dim = inputs.shape[2:]
        proto = preds[1][-1]
        preds = ops.non_max_suppression(preds[0], nc=len(model.names))

        masks_batch = []
        for i, pred in enumerate(preds):
            # no known object is detected
            if not len(pred):
                masks = None
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_dim, upsample=True)
            masks_batch.append(masks)

        # record end time
        count += inputs.shape[0]
        if count == expect_img_num:
            print(f"end: {time.monotonic()}")
    _sub = pygaps.Subscriber(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP, msg_handler)

    print("Ctrl+C to leave")
    signal.pause()

if __name__ == "__main__":
    main()