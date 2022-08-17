import argparse
import numpy as np
import time
import os

from utils.mot.detector_model import DetectorModel


def meas_sess(sess, n):
    print("Session created")
    inputs = [t for t in sess.inputs()]
    shapes = [i.shape for i in inputs]

    tensors = [np.zeros(shape, dtype=np.uint8) for shape in shapes]
    
    outputs = sess.run(tensors)

    outputs_desc = [o.desc for o in outputs]
    print("Inputs:", inputs)
    print("Outputs:", outputs_desc)

    deltas = []

    # warmup
    for _ in range(3):
        tensors = [np.random.randint(0, 255, size=shape, dtype=np.uint8) for shape in shapes]
        sess.run(tensors)

    for _ in range(n):
        tensors = [np.random.randint(0, 255, size=shape, dtype=np.uint8) for shape in shapes]
        t1 = time.time()
        sess.run(tensors)
        delta_ms = (time.time() - t1) * 1e3
        deltas.append(delta_ms)
        print(f"Inference took {delta_ms:.2f}ms")

    delta_med = np.median(deltas)
    print(f"Median inference time: {delta_med:.2f}ms")

    return deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512], help="yolo input width and height")
    args = parser.parse_args()

    batch_size = args.batch_size
    w, h = args.size

    # os.environ["NPU_PROFILER_PATH"] = "npu.json"

    det_model = DetectorModel(0, "models/yolov5m_warboy.yaml", "runs/train/bytetrack_mot20_5data/weights/best.pt", "furiosa", 
        input_size=(w, h), batch_size=batch_size, input_type="i8", input_format="hwc")

    sess = det_model.model.sess

    meas_sess(sess, args.n)

    sess.close()


if __name__ == "__main__":
    main()
