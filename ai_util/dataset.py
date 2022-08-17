from pathlib import Path
import cv2
import glob


def batch_iter(it, batch_size):
    if isinstance(it, list):
        it = iter(it)

    has_items = True
    while has_items:
        batch = []

        try:
            for _ in range(batch_size):
                batch.append(next(it))
        except StopIteration:
            has_items = False
        
        if len(batch) > 0:
            yield batch


class ImageDataset:
    def __init__(self, path, limit=None, start_idx=0, end_idx=None, frame_step=1, fps=30) -> None:
        self.path = path
        self.src = None
        self.next_idx = start_idx
        self.limit = limit
        self.frame_step = frame_step
        self.fps = fps
        self.length = None
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.open()  # to read frame count etc.
        if self.is_video():
            self.close()

    def is_video(self):
        return isinstance(self.path, (str, Path)) and Path(self.path).suffix in (".mp4", ".avi")

    def is_open(self):
        return self.src is not None

    def open(self):
        assert not (self.is_video() and self.is_open()), "Can lead to threading errors with opencv"

        if isinstance(self.path, list):
            self.src = self.path
        elif Path(self.path).is_dir():
            self.src = sorted(Path(self.path).glob("*"))
            assert len(self.src) > 0
        elif self.is_video():
            self.src = cv2.VideoCapture(str(self.path))
            assert self.src.isOpened()
        else:
            self.src = glob.glob(self.path)
            assert len(self.src) > 0

        self.fps = self.fps if isinstance(self.src, list) else self.src.get(cv2.CAP_PROP_FPS)
        self.length = len(self.src) if isinstance(self.src, list) else int(self.src.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self):
        if self.is_video():
            assert self.is_open()

            if not isinstance(self.src, list):
                self.src.release()
            self.src = None

            assert not self.is_open()

    # @staticmethod
    def load_image_by_key(self, key):
        assert self.is_open()

        if isinstance(self.src, list):
            return cv2.cvtColor(cv2.imread(str(self.src[key])), cv2.COLOR_BGR2RGB)
        else:
            # video_file, pos = key
            # assert Path(video_file) == self.path

            # cap = cv2.VideoCapture(video_file)
            old_pos = self.src.get(cv2.CAP_PROP_POS_FRAMES)
            self.src.set(cv2.CAP_PROP_POS_FRAMES, key)
            ret, img = self.src.read()
            
            if not ret:
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.src.set(cv2.CAP_PROP_POS_FRAMES, old_pos)

            return img

    def set_frame_idx(self, idx):
        self.next_idx = idx
        if not isinstance(self.src, list):
            self.src.set(cv2.CAP_PROP_POS_FRAMES, self.next_idx)

    def get_frame_idx(self):
        return self.next_idx

    def get_frame_count(self):
        return self.length

    def get_key_frame_count(self):
        return self.length / self.frame_step

    def get_fps(self):
        return self.fps

    def get_length_sec(self):
        return self.get_frame_count() / self.get_fps()

    def __len__(self):
        return self.length

    def __iter__(self):
        self.next_idx = self.start_idx
        self.open()
        return self

    def read(self):
        return self._read_next()[1]

    def get_next(self):
        return self._read_next()

    def _read_next(self):
        res = self._get_next()

        if res is None:
            self.close()
            raise StopIteration
        else:
            key, img = res

            if self.end_idx is not None and key >= self.end_idx:
                self.close()
                raise StopIteration

            return key, img

    def _get_next(self):
        assert self.is_open()
        assert self.next_idx is not None

        if self.limit is not None and self.next_idx >= self.limit:
            return None

        if isinstance(self.src, list):
            if self.next_idx >= len(self.src):
                return None

            key = self.next_idx
            assert Path(self.src[self.next_idx]).is_file(), f"{self.src[self.next_idx]} not found"
            img = cv2.cvtColor(cv2.imread(str(self.src[self.next_idx])), cv2.COLOR_BGR2RGB)
            self.next_idx += self.frame_step
        else:
            ret = False
            while not ret:
                if not self.src.isOpened():
                    return None

                self.src.grab()

                if self.next_idx % self.frame_step != 0:
                    ret = False
                else:
                    ret, img = self.src.retrieve()
                    # TODO: detect video end, isOpened not working
                    if not ret:
                        return None

                self.next_idx += 1
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            key = self.next_idx - 1

        return key, img

    def __next__(self):
        return self._read_next()