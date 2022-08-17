
import argparse
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from yolo_predictor.predictor import Yolov5Predictor, Tracker
from utils.inference_framework import PredictorBase


COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img

    tl = line_thickness or round(
        0.0004 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return img


def draw_bboxes(ori_img, bbox, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        if len(box) == 6:
            score = box[4]
            class_id = int(box[5])
            label = f"{class_id} ({score:.2f})"
        else:
            class_id = int(box[4])
            label = f"{class_id}"

        # box text and bar
        color = COLORS_10[class_id % len(COLORS_10)]
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/CrowdHuman/images/val/273271,128e30009deaa60c.jpg")
    parser.add_argument("--cfg", default="models/yolov5m_warboy.yaml")
    parser.add_argument("--weights", default="runs/train/bytetrack_mot20_5data/weights/best.pt")
    parser.add_argument("--input-size", nargs="+", default=(512, 512))
    parser.add_argument("--device", default="onnx_i8")
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args()

    batch_size = 4

    predictor = Yolov5Predictor(name=Path(args.cfg).stem, cfg=args.cfg, weights=args.weights, input_size=args.input_size, 
        batch_size=batch_size, pad_batch=True).to(args.device)

    if args.track:
        tracker = Tracker()
    else:
        tracker = None

    inp = Path(args.input)

    if inp.suffix == ".jpg":
        img = cv2.cvtColor(cv2.imread(str(inp)), cv2.COLOR_BGR2RGB)
        boxes = predictor(img)
        draw_bboxes(img, boxes[0])
        cv2.imwrite("out.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        out_dir = Path("out") / "video"
        out_dir.mkdir(parents=True, exist_ok=False)

        def _batch_iter():
            cap = cv2.VideoCapture(str(inp))

            batch = []
            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch.append(frame)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []

            cap.release()

            if len(batch) > 0:
                yield batch

        idx = 0
        for imgs in tqdm(_batch_iter()):
            boxes = predictor(imgs)
            
            for img, bb in zip(imgs, boxes):
                if tracker is not None:
                    bb = tracker(img, bb)
                draw_bboxes(img, bb)
                cv2.imwrite(str(out_dir / f"{idx:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                idx += 1


if __name__ == "__main__":
    main()
