import argparse
from multiprocessing import Process, Lock, Queue
import cv2
import time
import os

from utils.mot.video_input import OpenCVVideoRecord

from tqdm import tqdm


class MutliVideoRecord:
    def __init__(self, inputs, outputs, length) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.length = length
        self.fps = 30

    """
    def vid_read(self, src, qu):
        cap = cv2.VideoCapture(src)

        for i in range(self.length):
            ret = False

            while not ret:
                ret, img = cap.read()
                ts = time.time()

            qu.put((img, ts))

    def vid_write(self, out, qu):
        writer = OpenCVVideoRecord(out, fps=30, start_rec=True)
        out_ts = os.path.splitext(out)[0] + ".txt"

        timestamps = []

        for i in range(self.length):
            img, ts = qu.get()

            writer.update(img)
            timestamps.append(ts)

        writer.close()

        with open(out_ts, "w") as f:
            f.write("\n".join([str(ts) for ts in timestamps]))
    """

    def vid_rec(self, idx, src, dest):
        fps = self.fps
        cap = cv2.VideoCapture(src)
        writer = OpenCVVideoRecord(dest, fps=fps, start_rec=True)
        dest_ts = os.path.splitext(dest)[0] + ".txt"

        timestamps = []

        delta_target = 1 / fps

        it = range(self.length)

        if idx == 0:
            it = tqdm(it, total=self.length)

        for i in it:
            t1 = time.time()
            ret = False

            while not ret:
                ret, img = cap.read()
                ts = time.time()

            writer.update(img)
            timestamps.append(ts)

            delta = time.time() - t1
            delta_rem = delta_target - delta
            # if delta_rem > 0:
            #     time.sleep(delta_rem)


        writer.close()

        with open(dest_ts, "w") as f:
            f.write("\n".join([str(ts) for ts in timestamps]))

    def run(self):
        procs = [Process(target=self.vid_rec, args=(i, src, dest)) for i, (src, dest) in enumerate(zip(self.inputs, self.outputs))]

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=150)
    args = parser.parse_args()

    length = args.length

    input_src = [
        "rtsp://admin:!2gkakCVPR@192.168.1.101/profile2/media.smp",
        "rtsp://admin:!2gkakCVPR@192.168.1.102/profile2/media.smp",
        "rtsp://admin:!2gkakCVPR@192.168.1.104/profile2/media.smp",
        "rtsp://admin:!2gkakCVPR@192.168.1.105/profile2/media.smp",
    ]

    outputs = [
        f"out/rec/cam{out}.mp4" for out in [
            101,
            102,
            104,
            105
        ]
    ]

    inputs = [f"rtspsrc location={src} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true" for src in input_src]

    MutliVideoRecord(inputs, outputs, length).run()


if __name__ == "__main__":
    main()
