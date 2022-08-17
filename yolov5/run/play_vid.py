from typing import Any
import cv2
import numpy as np
import time
import os

from utils.mot.video_input import TSVideoStream

# need align method
# need move to latest time method


"""
class TSVideoStreamPoolInstance:
    def __init__(self, pool, idx) -> None:
        self.pool = pool
        self.idx = idx

    def __call__(self, *args, **kwargs):
        return self.pool.next_frame(self.idx, *args, **kwargs)


class TSVideoStreamPool:
    def __init__(self) -> None:
        self.streams = []
        self.has_started = False

    def _align(self):
        pass

    def next_frame(self, idx, *args: Any, **kwargs: Any) -> Any:
        # vs = TSVideoStream(*args, **kwargs)
        vs = self.streams[idx]
        return vs(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        vs = TSVideoStream(*args, **kwargs)
        self.streams.append(vs)
        return TSVideoStreamPoolInstance(self, len(self.streams) - 1)
"""


def stream_vid_realtime():
    file = "/Users/kevin/Downloads/rec/cam105.mp4"

    commom_start_time = TSVideoStream.get_common_start_time([file])

    stream = TSVideoStream(file, commom_start_time)
    # stream.move_start_time(7)
    # stream.start()

    while True:
        frame = stream()

        if frame is None:
            break

        cv2.imshow("out", frame)
        cv2.waitKey(1)


def read_cam_single():
    cap = cv2.VideoCapture("/Users/kevin/Documents/recv/gst-pipe/vid/cam103.mp4")

    last_time = cap.get(cv2.CAP_PROP_POS_MSEC)

    i = 0
    while cap.isOpened():
        print()
        ret, img = cap.read()
        cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        delta_ms = cur_time - last_time
        print(f"Delta: {delta_ms}")
        last_time = cur_time

        assert ret

        cv2.imshow("out", img)
        cv2.waitKey(1)

        i += 1

        if i > 10:
            break


def read_cam_multi():
    """
    caps = [
        cv2.VideoCapture("/Users/kevin/Documents/recv/gst-pipe/res/cam0.mp4"),
        cv2.VideoCapture("/Users/kevin/Documents/recv/gst-pipe/res/cam1.mp4")
    ]
    """

    caps = [
        cv2.VideoCapture("/Users/kevin/Documents/recv/gst-pipe/vid/cam103.mp4"),
        cv2.VideoCapture("/Users/kevin/Documents/recv/gst-pipe/vid/cam105.mp4")
    ]

    i = 0
    t1 = time.time()
    while caps[0].isOpened():
        time_ms = round((time.time() - t1) * 1e3)
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        print([cap.get(cv2.CAP_PROP_POS_MSEC) for cap in caps])
        imgs = [cap.read()[1] for cap in caps]

        img = np.concatenate(imgs, 1)

        # cv2.imwrite(f"/Users/kevin/Documents/recv/gst-pipe/res/frames/{i:04d}.jpg", img)

        cv2.imshow("out", img)
        cv2.waitKey(1)

        i += 1


def read_vid_stream():
    vid = "out/cam0.mp4"
    # cap = cv2.VideoCapture(f"filesrc location='{vid}' ! qtdemux ! queue ! h264parse ! avdec_h264 ! videoconvert ! appsink")
    cap = cv2.VideoCapture(f"filesrc location='{vid}' ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true")

    i = 0
    while cap.isOpened():
        print()
        ret, img = cap.read()

        assert ret

        cv2.imshow("out", img)
        cv2.waitKey(100)

def main():
    # read_cam_single()
    # read_vid_stream()
    # read_cam_multi()
    stream_vid_realtime()


if __name__ == "__main__":
    main()

