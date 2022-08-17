import time
import cv2
import glob
import os

import numpy as np


class CapInput:
    def __init__(self, src, size=None, frame_limit=None, framerate_limit=None, loop_video=False) -> None:
        try:
            src = int(src)
        except ValueError:
            pass

        if isinstance(src, str) and src.startswith("rtsp://"):
            src = f"rtspsrc location={src} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true"

        self.src = src

        # self.size = None
        self.cap = None
        self.frame_idx = 0
        self.size = tuple(size) if size is not None else None
        self.frame_limit = frame_limit
        self.loop_video = loop_video
        self.frame = None

        self.last_ts = None
        self.framerate_limit = framerate_limit

        self._open()

    def _open(self):
        cap = cv2.VideoCapture(self.src)
        
        if isinstance(self.src, int):
            if self.size is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

        cap_size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.size is not None:
            assert self.size == cap_size, f"Requested camera resolution differs from actual camera resolution ({self.size} != {cap_size})"
        else:
            self.size = cap_size

        self.cap = cap

        assert self.cap.isOpened(), f"CapInput failed to open '{self.src}'"

    def __call__(self):
        if self.last_ts is not None:
            delta = time.time() - self.last_ts

            if self.framerate_limit is not None:
                target_delta = 1 / self.framerate_limit
                delta_diff = target_delta - delta

                if delta_diff > 0:
                    time.sleep(delta_diff)

        self.last_ts = time.time()

        if self.frame_limit is not None and self.frame_idx >= self.frame_limit:
            return None
            # return np.zeros_like(self.frame)
            # return self.frame

        ret = False

        while not ret:
            if not self.cap.isOpened():
                self.close()
                frame = None
                break

            ret, frame = self.cap.read()

            if self.loop_video and not ret:
                break

        if not ret:
            frame = None

        if frame is not None:
            if self.size is not None:
                assert frame.shape[:2] == (self.size[1], self.size[0]), f"Capture does not support size {self.size}"

            self.frame_idx += 1
        else:
            if self.loop_video:
                print("loop")
                self._open()
                return self.__call__()

        self.frame = frame

        return frame

    def is_open(self):
        return self.cap.isOpened() and not (self.frame_limit is not None and self.frame_idx >= self.frame_limit)

    def close(self):
        self.cap.release()


# timestamped video stream
class TSVideoStream:
    def __init__(self, src, start_ts, start_barrier=None) -> None:
        src_ts = os.path.splitext(src)[0] + ".txt"
        self.timestamps = np.loadtxt(src_ts)
        self.ts_start = None
        self.frame_idx = 0
        self.cap = None
        self.src = src
        self.start_barrier = start_barrier

        self.first_frame_ts = self.timestamps[0]
        self.timestamps -= self.first_frame_ts

        self.cap = cv2.VideoCapture(self.src)
        assert self.cap.isOpened()

        self.set_start_time(start_ts)

    @staticmethod
    def get_common_start_time(srcs):
        timestamps = [TSVideoStream._load_ts(src)[0] for src in srcs]
        ts = max(timestamps)
        return ts

    @staticmethod
    def _load_ts(src):
        src_ts = os.path.splitext(src)[0] + ".txt"
        return np.loadtxt(src_ts) if os.path.isfile(src_ts) else None

    @staticmethod
    def is_ts_video_stream(src):
        return isinstance(src, str) and os.path.isfile(src) and TSVideoStream._load_ts(src) is not None

    def start(self):
        # optionally wait for other streams to start
        if self.start_barrier is not None:
            self.start_barrier.wait()

        self.ts_start = time.time()

    def is_open(self):
        return self.cap is not None and self.cap.isOpened()

    def set_start_time(self, start_time):
        self.move_start_time(start_time - self.first_frame_ts)

    def move_start_time(self, delta_time):
        assert self.ts_start is None
        assert delta_time >= 0, "Cannot go back"

        self.first_frame_ts += delta_time
        self.timestamps -= delta_time

        is_long_enough = self._seek_to(0)
        assert is_long_enough, f"video out of frames"

    def _seek_to(self, ts):
        while self.timestamps[self.frame_idx] <= ts:
            if self.frame_idx + 1 >= len(self.timestamps):
                return False

            self.cap.grab()
            self.frame_idx += 1

        return True

    def __call__(self):
        if self.ts_start is None:
            self.start()
        ts = time.time() - self.ts_start

        if not self._seek_to(ts):
            return None 

        ret, frame = self.cap.retrieve()
        assert ret

        return frame


class ImageInput:
    def __init__(self, path, frame_limit=None) -> None:
        self.files = sorted(glob.glob(os.path.join(path, "*")))
        self.frame_idx = 0
        self.frame_limit = frame_limit

    def __call__(self):
        if self.frame_idx >= len(self.files):
            return None

        if self.frame_limit is not None and self.frame_idx >= self.frame_limit:
            self.close()
            return None

        img = cv2.imread(self.files[self.frame_idx])
        self.frame_idx += 1

        return img

    def close(self):
        pass


class VideoInput:
    def __init__(self, src, *args, **kwargs) -> None:
        if os.path.isdir(src):
            inp = ImageInput(src, *args, **kwargs)
        else:
            inp = CapInput(src, *args, **kwargs)

        self.inp = inp

    @property
    def src(self):
        return self.inp.src

    @property
    def size(self):
        return self.inp.size

    def __call__(self):
        return self.inp() 

    def close(self):
        self.inp.close()

    def is_open(self):
        return self.inp.is_open()



class OpenCVVideoRecord:
    def __init__(self, filename, fps=10, start_rec=False, is_rgb=False) -> None:
        self.filename = filename
        self.writer = None
        self.fps = fps
        self.start_rec = start_rec
        self.is_rgb = is_rgb
        self.record_completed = False
        self.size = None

    def toggle_record(self, size):
        if self.filename is not None:
            if self.writer is None:
                # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if not os.path.isdir(os.path.dirname(self.filename)):
                    os.makedirs(os.path.dirname(self.filename))

                fourcc = cv2.VideoWriter_fourcc(*'VP80') if self.filename.endswith(".webm") else cv2.VideoWriter_fourcc(*"XVID")
                self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, tuple(size))
                self.size = size
                print("Start record")
            else:
                self.writer.release()
                self.writer = None
                self.record_completed = True
                self.size = None
                print("End record")

    def is_recording(self):
        return self.start_rec or self.writer is not None

    def update(self, img, key=None):
        if key is not None:
            key = key & 0xff

        if key == ord("r") or self.start_rec:
            self.start_rec = False
            self.toggle_record((img.shape[1], img.shape[0]))

        if self.filename is not None:
            if self.writer is not None:
                if self.is_rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.writer.write(img)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("End record")
