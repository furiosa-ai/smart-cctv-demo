import cv2
import os
import glob


class ImageRecord:
    def __init__(self, filename, start_rec=False) -> None:
        self.filename = filename
        self.idx = 0
        self.on_record = start_rec

        if not os.path.isdir(self.filename):
            os.makedirs(self.filename)

    def toggle_record(self):
        self.on_record = not self.on_record

    def is_recording(self):
        return self.on_record

    def update(self, img, key):
        if key is not None:
            key = key & 0xff

        if key == ord("r"):
            self.toggle_record()

        if self.on_record:
            out_file = os.path.join(self.filename, f"{self.idx:04d}.jpg")
            self.idx += 1
            # cv2.imwrite(out_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_file, img)

    def close(self):
        pass


class OpenCVVideoRecord:
    def __init__(self, filename, fps=10, start_rec=False, is_rgb=False) -> None:
        self.filename = filename
        self.writer = None
        self.fps = fps
        self.start_rec = start_rec
        self.is_rgb = is_rgb

    def toggle_record(self, size):
        if self.filename is not None:
            if self.writer is None:
                # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fourcc = cv2.VideoWriter_fourcc(*'VP80') if self.filename.endswith(".webm") else cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, tuple(size))
                print("Start record")
            else:
                self.writer.release()
                self.writer = None
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


class VideoInput:
    def __init__(self, filename) -> None:
        try:
            filename = int(filename)
        except ValueError:
            pass

        self.index = 0
        self.filename = filename

        if isinstance(filename, int):
            self.cap = cv2.VideoCapture(filename)
        elif filename.endswith(".txt"):
            root_dir = os.path.dirname(filename)
            with open(filename, "r") as f:
                files = [l.rstrip() for l in f.readlines()]

            self.cap = [os.path.join(root_dir, f) for f in files]
        elif os.path.isdir(filename):
            self.cap = sorted(glob.glob(os.path.join(filename, "*.jpg")))
        else:
            self.cap = cv2.VideoCapture(filename)

    def read(self):
        if isinstance(self.cap, cv2.VideoCapture):
            ret, img = self.cap.read()

            if not ret:
                img = None
        else:
            img = cv2.imread(self.cap[self.index])
            self.index += 1

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def is_open(self):
        if isinstance(self.cap, cv2.VideoCapture):
            return self.cap.isOpened()
        else:
            return self.index < len(self.cap)

    def get_frame_count(self):
        if isinstance(self.cap, cv2.VideoCapture):
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not isinstance(self.filename, int) else None
        else:
            return len(self.cap)

    def close(self):
        if isinstance(self.cap, cv2.VideoCapture):
            self.cap.release()
        self.cap = None
