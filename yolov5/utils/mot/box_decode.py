import numpy as np


def _box_decode_feat(anchors, stride, conf_thres, feat, out_boxes_batched):
    bs, na, ny, nx, no = feat.shape  # no = 4 + nc
    nc = no - 5
    assert nc == 1  # single class only for now

    cls = 0  # usually need to compute as "cls = argmax([b, a, y, x, 5:])"

    for b in range(bs):
        for a in range(na):
            ax = anchors[a, 0] * stride
            ay = anchors[a, 1] * stride

            for y in range(ny):
                for x in range(nx):
                    conf = feat[b, a, y, x, 4]

                    if conf > conf_thres:
                        # usually need to compute final confidence as "conf *= max(feat[b, a, y, x, 5:])"" 
                        # and check again "conf > conf_thres"

                        # (feat[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                        # (feat[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        bx = feat[b, a, y, x, 0]
                        by = feat[b, a, y, x, 1]
                        bw = feat[b, a, y, x, 2]
                        bh = feat[b, a, y, x, 3]

                        bx = (bx * 2 - 0.5 + x) * stride
                        by = (by * 2 - 0.5 + y) * stride

                        bw *= 2
                        bh *= 2
                        bw = (bw * bw) * ax
                        bh = (bh * bh) * ay
                    
                        bw_half = 0.5 * bw
                        bh_half = 0.5 * bh

                        bx1 = bx - bw_half
                        bx2 = bx + bw_half
                        by1 = by - bh_half
                        by2 = by + bh_half

                        out_boxes_batched[b].append([bx1, by1, bx2, by2, conf, cls])


def box_decode(anchors, stride, conf_thres, feats):
    # no, na = anchors.shape[:2]
    # bs, _, ny, nx = feats[0].shape
    bs = feats[0].shape[0]

    out_boxes_batched = [[] for _ in range(bs)]

    for l, feat in enumerate(feats):
        _box_decode_feat(anchors[l], stride[l], conf_thres, feat, out_boxes_batched)

    out_boxes_batched = [np.array(o, dtype=np.float32) for o in out_boxes_batched]

    return out_boxes_batched
