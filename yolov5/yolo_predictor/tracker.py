

from bytetrack.deploy.cbytetrack import CByteTrack


class Tracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8) -> None:
        self.tracker = CByteTrack()

    def __repr__(self) -> str:
        return "Tracker(" + ", ".join([f"{n}: {len(v)}" for n, v in zip(
            ("Tracked", "Lost", "Removed"),
            (self.tracker.tracked_stracks, self.tracker.lost_stracks, self.tracker.removed_stracks)
        )]) + ")"

    def __call__(self, boxes):
        # does not accept batch
        tracker_input = boxes[:, :5]

        pred = self.tracker.update(tracker_input)

        return pred
