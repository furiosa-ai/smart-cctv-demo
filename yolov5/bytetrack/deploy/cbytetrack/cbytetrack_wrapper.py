import ctypes
import numpy as np
import os
import torch
import platform


if platform.uname()[0] == "Windows":
    _lib_ext = "dll"
elif platform.uname()[0] == "Linux":
    _lib_ext = "so"
else:
    _lib_ext = "dylib"

_clib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'build', f'libbytetrack.{_lib_ext}'))


def _init():
    vp = ctypes.c_void_p
    u32 = ctypes.c_uint32
    f32 = ctypes.c_float

    u32p = np.ctypeslib.ndpointer(dtype=u32, ndim=1, flags='C_CONTIGUOUS')
    f32p = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
    # f32p_2d = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
    # f32p_3d = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='C_CONTIGUOUS')
    # f32p_5d = np.ctypeslib.ndpointer(dtype=np.float32, ndim=5, flags='C_CONTIGUOUS')

    _clib.ByteTrackNew.argtypes = []
    _clib.ByteTrackNew.restype = vp

    _clib.ByteTrackDelete.argtypes = [vp]
    _clib.ByteTrackDelete.restype = None

    _clib.ByteTrackUpdate.argtypes = [
        vp,
        f32p,
        # u32p,
        u32,
        f32p,
    ]
    _clib.ByteTrackUpdate.restype = u32

"""
def _box_decode_feat(anchors, stride, conf_thres, max_boxes, feat, out_batch, out_batch_pos):
    bs, na, ny, nx, no = feat.shape

    _clib.box_decode_feat(anchors.reshape(-1), na, stride, conf_thres, max_boxes, feat.reshape(-1), bs, ny, nx, no, out_batch.reshape(-1), out_batch_pos)


def box_decode(anchors, stride, conf_thres, feats):
    bs = feats[0].shape[0]
    max_boxes = int(1e4)

    out_batch = np.empty((bs, max_boxes, 6), dtype=np.float32)
    out_batch_pos = np.zeros(bs, dtype=np.uint32)

    for l, feat in enumerate(feats):
        _box_decode_feat(anchors[l], stride[l], conf_thres, max_boxes, feat, out_batch, out_batch_pos)

    out_boxes_batched = [boxes[:(pos // 6)] for boxes, pos in zip(out_batch, out_batch_pos)]

    return out_boxes_batched
"""


class CByteTrack:
    def __init__(self) -> None:
        self.obj = _clib.ByteTrackNew()
        self.buffer = np.zeros((int(1e5), 5), dtype=np.float32)

    def update(self, boxes):
        assert boxes.shape[1] == 5, "Need xyxyp format"
        # boxes_conf, classes = boxes[:, :4].astype(np.float32), boxes[:, 4].astype(np.uint32)
        if isinstance(boxes, torch.Tensor):
            boxes_conf = boxes.type(torch.float32).numpy()
        else:
            boxes_conf = boxes.astype(np.float32)
        num_boxes = boxes.shape[0]
        num_tracks = _clib.ByteTrackUpdate(self.obj, boxes_conf.reshape(-1), num_boxes, self.buffer.reshape(-1))
        boxes_track = self.buffer[:num_tracks].copy()
        return boxes_track

    def close(self):
        if self.obj is not None:
            _clib.ByteTrackDelete(self.obj)
            self.obj = None

    def __del__(self):
        self.close()


_init()
