import sys
import numpy as np
from types import SimpleNamespace

from utils.logging import log_func

sys.path.append("bytetrack")
from bytetrack.deploy.cbytetrack import CByteTrack


class Tracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, max_track_count=None, backend="c") -> None:
        assert backend in ("c", "python")
        self.use_cbyte_track = True

        if backend == "c":
            self.tracker = CByteTrack()
        elif backend == "python":
            from bytetrack.yolox.tracker.byte_tracker import BYTETracker
            tracker_args = {
                "track_thresh": track_thresh,
                "track_buffer": track_buffer,
                "match_thresh": match_thresh,
                "mot20": False
            }
            self.tracker = BYTETracker(SimpleNamespace(**tracker_args))

        self.max_track_count = None

    def __repr__(self) -> str:
        return "Tracker(" + ", ".join([f"{n}: {len(v)}" for n, v in zip(
            ("Tracked", "Lost", "Removed"),
            (self.tracker.tracked_stracks, self.tracker.lost_stracks, self.tracker.removed_stracks)
        )]) + ")"

    """
    def limit_tracker(self):
        if self.max_track_count is not None and self.max_track_count > 0 and len(self.tracker.tracked_stracks) > self.max_track_count:
            rem_count = len(self.tracker.tracked_stracks) - self.max_track_count

            track_idx = 0
            track_len = 0

            track_rem = set([])

            while len(track_rem) < rem_count:
                if track_idx >= len(self.tracker.tracked_stracks):
                    # restart and remove next longest tracks
                    track_idx = 0
                    track_len += 1

                if self.tracker.tracked_stracks[track_idx].tracklet_len == track_len:
                    track_rem.add(track_idx)

                track_idx += 1

            self.tracker.tracked_stracks = [t for i, t in enumerate(self.tracker.tracked_stracks) if i not in track_rem]
            assert len(self.tracker.tracked_stracks) <= self.max_track_count
    """

    @log_func
    def __call__(self, img, boxes):
        # does not accept batch
        tracker_input = boxes[:, :5]

        input_size = img.shape[:2]

        if self.use_cbyte_track:
            pred = self.tracker.update(tracker_input)
        else:
            # online_targets = tracker.update(tracker_input, origin_size, input_size)
            online_targets = self.tracker.update(tracker_input, input_size, input_size)
            # self.limit_tracker()
            # online_targets = [track for track in self.tracker.tracked_stracks if track.is_activated]

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # override pred, replace cls index with track id
            # TODO: remove loop
            pred = []
            for t in online_targets:
                x1, y1, w, h = t.tlwh
                x2, y2 = x1 + w, y1 + h
                track_id = t.track_id

                pred.append([x1, y1, x2, y2, track_id])
            # pred = [torch.tensor(pred)]
            pred = np.array(pred, dtype=np.float32)

        # print(self)

        return pred
