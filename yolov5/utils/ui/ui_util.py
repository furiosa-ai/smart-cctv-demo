
import numpy as np

from utils.util import dump_args, load_args
from utils.ui.cui_util import create_plot_data as ccreate_plot_data


def create_plot_data(color_table, lines, line_ids):
    # dump_args("create_plot_data", dict(color_table=color_table, lines=lines, line_ids=line_ids))
    # lines: list of nx2 float64 arrays
    # array of ints

    # filter nan
    lines = [line[~np.isnan(line[:, 0])] for line in lines]

    # color = [self.parent()._id_to_color(line_id) for line_id in line_ids]
    color = color_table[line_ids % len(color_table)]

    pos = np.concatenate(lines)
    color = np.concatenate([([c] * len(line)) for line, c in zip(lines, color)])

    con_a = []
    con_b = []

    i = 0
    for line in lines:
        l = len(line)
        con_a.append(np.arange(i+0, l+i-1))
        con_b.append(np.arange(i+1, l+i-0))
        i += l

    con = np.stack([
        np.concatenate(con_a),
        np.concatenate(con_b)
    ], 1)

    # min_pos, max_pos = np.min(pos, 0), np.max(pos, 0)

    # pos = (pos - min_pos[None]) / (max_pos - min_pos)[None]

    return {
        "pos": pos,
        "connect": con,
        "color": color
    }


def create_plot_data_loop(color_table, lines, line_ids):
    num_lines = len(lines)
    num_colors = len(color_table)

    max_buf = int(1e5)

    pos = np.empty((max_buf, 2), dtype=np.float32)
    con = np.empty((max_buf, 2), dtype=np.int64)
    color = np.empty((max_buf, 3), dtype=np.float32)

    pi = 0
    ci = 0
    for line_idx in range(num_lines):
        line = lines[line_idx]
        line_id = line_ids[line_idx]

        c = color_table[line_id % num_colors]
        is_first = True

        for point_idx in range(line.shape[0]):
            x, y = line[point_idx]

            if x != np.nan:
                assert pi < max_buf, "Buffer too small"

                pos[pi][0] = x
                pos[pi][1] = y

                color[pi][0] = c[0]
                color[pi][1] = c[1]
                color[pi][2] = c[2]

                if not is_first:
                    con[ci][0] = pi - 1
                    con[ci][1] = pi
                    ci += 1

                pi += 1
                is_first = False

    pos = pos[:pi]
    con = con[:ci]
    color = color[:pi]

    return {
        "pos": pos,
        "connect": con,
        "color": color
    }


def create_box_data(color_table, boxes):
    # boxes: relative xyxy + trackid
    num_boxes = len(boxes)

    xyxy = boxes[:, :4]
    color_boxes = color_table[boxes[:, 4].astype(int) % len(color_table)]

    pos = xyxy[:, [
        0, 1,  # tl
        2, 1,  # tr
        2, 3,  # br
        0, 3,  # bl
    ]].reshape(4 * num_boxes, 2)

    # 01   01 12 23 30
    # 32

    con_single = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int64)
    con = con_single[None, :] + np.arange(0, 4 * num_boxes, 4, dtype=np.int64)[:, None]
    con = con.reshape(num_boxes * 4, 2)

    color = np.repeat(color_boxes, 4, axis=0)

    return {
        "pos": pos,
        "connect": con,
        "color": color
    }


def _test():
    kwargs = load_args("create_plot_data")

    out1 = create_plot_data(**kwargs)
    # out2 = create_plot_data_loop(**kwargs)
    out2 = ccreate_plot_data(**kwargs)

    for key in out1.keys():
        arr1, arr2 = out1[key], out2[key]

        assert np.allclose(arr1, arr2)

    print("done")


if __name__ == "__main__":
    _test()
