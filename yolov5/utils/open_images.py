from cProfile import label
import os
from typing import OrderedDict
import numpy as np
import tqdm


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


class OpenImagesData:
    def __init__(self, root_path, mode, class_filter=None) -> None:
        root_path = os.path.join(root_path, "train" if mode == "train" else "validation")

        with open(os.path.join(root_path, "labels/detections.csv"), "r") as f:
            lines = [l.rstrip().split(",") for l in tqdm.tqdm(f.readlines(), desc="Reading detections")]

        with open(os.path.join(root_path, "metadata/classes.csv"), "r") as f:
            class_map = [l.rstrip().split(",") for l in f.readlines()]
        class_map = {k: v for k, v in class_map}

        keys, lines = lines[0], lines[1:]
        idx = {key: i for i, key in enumerate(keys)}

        boxes_xyxy = OrderedDict()
        class_idx = OrderedDict()
        label_names = OrderedDict()

        if class_filter is not None:
            class_filter = set([next(iter(k for k, v in class_map.items() if v in cls)) for cls in class_filter])

        for vals in tqdm.tqdm(lines, "Processing data"):
            name = vals[idx["ImageID"]]
            label_name = vals[idx["LabelName"]]

            if class_filter is None or label_name in class_filter:
                img_file = os.path.join(root_path, "data", f"{name}.jpg")

                if os.path.isfile(img_file):
                    xyxy = [vals[idx[k]] for k in ("XMin","YMin","XMax","YMax")]

                    if label_name not in label_names:
                        label_names[label_name] = len(label_names)

                    if name not in boxes_xyxy:
                        boxes_xyxy[name] = []
                        class_idx[name] = []

                    class_idx[name].append(label_names[label_name])
                    boxes_xyxy[name].append(xyxy)

        names = list(class_idx.keys())
        class_idx = [np.array(v, dtype=int) for v in class_idx.values()]
        boxes_xyxy = [np.array(v, dtype=float) for v in boxes_xyxy.values()]
        boxes_xywh = [xyxy2xywh(v) for v in boxes_xyxy]

        # assert os.path.isfile(img_files[0])
        # img_files = [f for f in img_files if os.path.isfile(f)]

        class_names = []

        for label_name, i in label_names.items():
            assert i == len(class_names)
            class_names.append(class_map[label_name])

        self.names = names
        self.img_files = [os.path.join(root_path, "data", f"{n}.jpg") for n in self.names]
        self.boxes = boxes_xywh
        self.class_idx = class_idx
        self.class_names = class_names
        self.mode = mode

    def save_labels_as_yolo(self, out_dir):
        img_list_file = os.path.join(out_dir, f"{self.mode}.txt")
        out_dir_labels = os.path.join(out_dir, self.mode, "labels")

        assert not os.path.exists(img_list_file)

        if not os.path.exists(out_dir_labels):
            os.makedirs(out_dir_labels)

        for name, cls, box in zip(self.names, self.class_idx, self.boxes):
            cont = "\n".join([f"{c} {x} {y} {w} {h}" for c, (x, y, w, h) in zip(cls, box)])
            label_file = os.path.join(out_dir_labels, f"{name}.txt")

            with open(label_file, "w") as f:
                f.write(cont)

        img_list = "\n".join([f"./{self.mode}/images/{n}.jpg" for n in self.names])

        with open(img_list_file, "w") as f:
            f.write(img_list)


def _test():
    data = OpenImagesData("../data/open-images-v6", "train", class_filter=("Person",))

    data.save_labels_as_yolo("../data/open-images-v6/yolo")


if __name__ == "__main__":
    _test()
