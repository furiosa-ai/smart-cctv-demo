import numpy as np
import glob
import tqdm
import os
import random

class_names = set([])

def conv_boxes(filename):
    iw, ih = 1936, 1464
    boxes = []

    classes = []

    with open(filename, "r") as f:
        lines = [l.rstrip() for l in f.readlines()]
    for line in lines:
        vals = line.split(" ")

        class_names.add(vals[0])

        classes.append(vals[0])
        boxes.append(vals[1:])

    boxes = np.array(boxes, dtype=float)

    boxes[:, 0::2] /= iw - 1
    boxes[:, 1::2] /= ih - 1

    # xyxy to xywh
    boxes[:, 2:4] -= boxes[:, 0:2]
    boxes[:, 0:2] += boxes[:, 2:4] / 2

    out = "\n".join([" ".join([c] + [str(v) for v in b]) for c, b in zip(classes, boxes)])

    return out


def conv_labels():
    src_dir = "/mnt/sda/Documents/projects/data/car/main/txt/"
    dst_dir = "/mnt/sda/Documents/projects/data/car/yolov5/labels"

    files = glob.glob(src_dir + "*")

    img_boxes = [conv_boxes(f) for f in tqdm.tqdm(files)]

    print(class_names)

    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    dst_files = [os.path.join(dst_dir, f"{i}_origin.txt") for i in file_ids]

    for cont, filename in tqdm.tqdm(zip(img_boxes, dst_files), total=len(img_boxes)):
        with open(filename, "w") as f:
            f.write(cont)


def create_split():
    root_dir = "/mnt/sda/Documents/projects/data/car/yolov5/"

    imgs = glob.glob(root_dir + "images/*")
    random.shuffle(imgs)

    imgs = ["./" + os.path.relpath(img, root_dir) for img in imgs]

    s = round(len(imgs) * 0.8)

    splits = [
        ("train", imgs[:s]),
        ("val", imgs[s:])
    ]

    for name, split in splits:
        fn = os.path.join(root_dir, f"{name}.txt")
        out = "\n".join(split)

        with open(fn, "w") as f:
            f.write(out)


def main():
    create_split()


if __name__ == "__main__":
    main()