#!/usr/bin/env python3
import os
import sys

from ultralytics import YOLO

DEVICE = "cuda"

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} IMG_DIR REQ_NUM [MIN_FACE_NUM_PER_IMG=1] [MODEL_PATH=./yolo11n-face.pt]")
        exit(1)

    img_dir = sys.argv[1]
    req_num = int(sys.argv[2])
    min_face_num = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model_path = sys.argv[4] if len(sys.argv) > 4 else "yolo11n-face.pt"

    model = YOLO(model_path).to(DEVICE)

    filenames = []
    count = 0
    for i, p in enumerate(os.listdir(img_dir)):
        print(f"Checking {i + 1}th image...")

        res = model(f"{img_dir}/{p}")
        
        if res[0].boxes.shape[0] >= min_face_num:
            filenames.append(p)
            count += 1

        if count == req_num:
            break

    if count < req_num:
        print(f"There are fewer than {req_num} images with faces.")

    output_name = f"paths_to_images_with_at_least_{min_face_num}_faces.txt"
    with open(output_name, "w") as f:
        for filename in filenames:
            f.write(f"{img_dir}/{filename}\n")
    print(f"{output_name} is generated")

if __name__ == "__main__":
    main()
