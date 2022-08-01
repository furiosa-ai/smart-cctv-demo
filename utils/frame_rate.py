import time


class FrameRateSync:
    def __init__(self, target_fps=None) -> None:
        self.target_fps = target_fps
        self.t1 = None

    def __call__(self, target_fps=None):
        target_fps = target_fps if target_fps is not None else self.target_fps

        target_delta = 1 / target_fps

        if self.t1 is not None:
            cur_delta = time.time() - self.t1
            delta_diff = target_delta - cur_delta
            if delta_diff > 0:
                time.sleep(delta_diff)

        self.t1 = time.time()
