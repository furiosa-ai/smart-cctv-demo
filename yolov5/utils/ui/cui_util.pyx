# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport isnan, NAN, INFINITY, fabs, sqrt

np.import_array()



cdef unsigned int _max_buf = int(1e6)


cdef int _imax(int a, int b):
    return a if a > b else b


def apply_line_color_gradient(dict plot_data, float alpha_min=0.8, float alpha_max=1.0):
    cdef np.ndarray color_np
    cdef unsigned int num_lines, ci, li, pi, line_length
    cdef float alpha_diff, weight, alpha, line_norm
    cdef long[:] line_lengths
    cdef float[:] color

    alpha_diff = alpha_max - alpha_min

    line_lengths = plot_data["line_lengths"]
    color_np = plot_data["color"].copy()
    color = color_np.reshape(-1)

    num_lines = line_lengths.shape[0]

    ci = 0
    for li in range(num_lines):
        line_length = line_lengths[li]
        line_norm = _imax(line_length - 1, 1)

        for pi in range(line_length):
            weight = pi / line_norm
            alpha = alpha_min + weight * alpha_diff

            color[ci + 0] *= alpha
            color[ci + 1] *= alpha
            color[ci + 2] *= alpha

            ci += 3

    return color_np


def create_plot_data(float[:, :] color_table, list lines, unsigned int[:] line_ids):
    cdef unsigned int num_lines, num_colors, line_idx, line_id, \
        point_idx, pi, ci, li, dim, cur_line_length, cur_num_points
    cdef float x, y

    cdef np.ndarray pos_np, color_np, con_np, line_ids_out_np, line_lengths_np, point_age_np

    cdef double[:, :] pos
    cdef float[:, :] color
    cdef long[:, :] con
    cdef long[:] line_ids_out, line_lengths, point_age

    cdef double[:, :] line
    cdef float[:] c

    num_lines = len(lines)
    num_colors = len(color_table)

    pos_np = np.empty((_max_buf, lines[0].shape[1] if len(lines) > 0 else 3), dtype=np.float64)
    con_np = np.empty((_max_buf, 2), dtype=np.int64)
    color_np = np.empty((_max_buf, 3), dtype=np.float32)
    line_ids_out_np = np.zeros(num_lines, dtype=np.int64)
    line_lengths_np = np.zeros(num_lines, dtype=np.int64)  # max num_lines, empty lines will be skipped
    point_age_np = np.empty(_max_buf, dtype=np.int64)

    pos = pos_np
    con = con_np
    color = color_np
    line_ids_out = line_ids_out_np
    line_lengths = line_lengths_np
    point_age = point_age_np

    pi = 0
    ci = 0
    li = 0
    for line_idx in range(num_lines):
        line = lines[line_idx]
        line_id = line_ids[line_idx]

        c = color_table[line_id % num_colors]

        if 2 * pi > _max_buf:
            assert False, "Critical number of plots reached"

        cur_line_length = 0
        cur_num_points = line.shape[0]

        for point_idx in range(cur_num_points):
        # for point_idx in range(cur_num_points - 1, -1, -1):
            if not isnan(line[point_idx][0]):
                # assert pi < _max_buf, "Buffer too small"

                # pos[pi][0] = x
                # pos[pi][1] = y

                # generalizes to 2d and 3d
                pos[pi] = line[point_idx]

                color[pi][0] = c[0]
                color[pi][1] = c[1]
                color[pi][2] = c[2]

                point_age[pi] = cur_num_points - (point_idx + 1)

                if cur_line_length > 0:
                    con[ci][0] = pi - 1
                    con[ci][1] = pi
                    ci += 1

                pi += 1
                cur_line_length += 1

        # ignore empty lines
        if cur_line_length > 0:
            line_lengths[li] = cur_line_length
            line_ids_out[li] = line_id
            li += 1

    pos_np = pos_np[:pi]
    con_np = con_np[:ci]
    color_np = color_np[:pi]
    point_age_np = point_age_np[:pi]
    line_ids_out_np = line_ids_out_np[:li]
    line_lengths_np = line_lengths_np[:li]

    return {
        "line_ids": line_ids_out_np,
        "line_lengths": line_lengths_np,
        "pos": pos_np,
        "connect": con_np,
        "color": color_np,
        "point_age": point_age_np
    }
