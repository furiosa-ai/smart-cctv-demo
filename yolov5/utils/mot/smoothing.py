import math
import numpy as np

from utils.logging import log_func


def _smooth_damp(current, target, current_velocity, smooth_time, max_speed=np.inf, delta_time=1, handle_overshooting=True):
    # current = np.array(current)
    # target = np.array(target)
    # current_velocity = np.array(current_velocity)

    # Based on Game Programming Gems 4 Chapter 1.10
    smooth_time = max(0.0001, smooth_time)
    omega = 2 / smooth_time

    x = omega * delta_time
    exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x)

    change = current - target
    originalTo = target

    # Clamp maximum speed
    if max_speed != np.inf:
        maxChange = max_speed * smooth_time

        maxChangeSq = maxChange * maxChange
        sqrmag = np.dot(change, change)
        if sqrmag > maxChangeSq:
            mag = math.sqrt(sqrmag)
            change = change / mag * maxChange

    target = current - change

    temp = (current_velocity + omega * change) * delta_time

    current_velocity = (current_velocity - omega * temp) * exp

    output = target + (change + temp) * exp

    # Prevent overshooting
    if handle_overshooting:
        origMinusCurrent = originalTo - current
        outMinusOrig = output - originalTo

        if current.ndim == 1:
            if np.dot(origMinusCurrent, outMinusOrig) > 0:
                output = originalTo

                current_velocity = (output - originalTo) / delta_time
        else:
            overshoot_mask = np.einsum("...i,...i->...", origMinusCurrent, outMinusOrig) > 0
            output[overshoot_mask] = originalTo[overshoot_mask]
            current_velocity[overshoot_mask] = 0

    return output, current_velocity


class IdArray:
    def __init__(self, ids=None, data=None, max_size=int(1e6)) -> None:
        if isinstance(ids, (tuple, list, np.ndarray)):
            ids = {id: i for i, id in enumerate(ids)}
        assert ids is None or isinstance(ids, dict)

        self.ids = ids if ids is not None else {}
        self.arr = data
        self.cur_idx = 0
        self.max_size = max_size

    def clear(self):
        self.ids = {}
        self.arr = None

    def get(self, id, default):
        if id in self.ids:
            return self.arr[self.ids[id]]
        else:
            return default()

    def __setitem__(self, id, x):
        if id not in self.ids:
            self.ids[id] = self.cur_idx
            self.cur_idx += 1
        idx = self.ids[id]

        if self.arr is None:
            self.arr = np.zeros((self.max_size, *x.shape), dtype=x.dtype)
        self.arr[idx] = x

    def data(self):
        return self.arr


class SmoothingBase:
    def __init__(self, use_batch_proc=True) -> None:
        self.state = IdArray() if use_batch_proc else {}
        self.use_batch_proc = use_batch_proc

    @log_func
    def __call__(self, data, ids=None):
        if ids is None:
            if isinstance(self.state, dict):
                self.state = None

            y, self.state = self._update(self.state, data)
            return y
        else:
            return self.update_batch(data, ids) if self.use_batch_proc else self.update_loop(data, ids)

    def _update(self, cur_state, x):
        raise NotImplementedError()

    def clear(self):
        self.state.clear()

    def update_loop(self, data, ids):
        # raise NotImplementedError()

        new_state = {}
        out = np.zeros_like(data)

        for i, (id, x) in enumerate(zip(ids, data)):
            id = int(id)

            out[i], new_state[id] = self._update(self.state.get(id, None), x)

        self.state = new_state

        return out

    def update_batch(self, data, ids):
        if len(data) > 0:
            ids = ids.astype(int)
            # raise NotImplementedError()

            # order last state to match current data ids
            last_state = IdArray(max_size=data.shape[0])
            for (id, x) in zip(ids, data):
                last_state[id] = self.state.get(id, lambda: self._new_state(x))

            if data.shape[0] > 0:
                new_state_data, out = self._update_batch(last_state.data(), data)
            else:
                new_state_data, out = None, data

            new_state = IdArray(ids, new_state_data, max_size=data.shape[0])
        else:
            out = data
            new_state = IdArray()

        self.state = new_state

        return out


class SmoothingNo(SmoothingBase):
    def __init__(self) -> None:
        super().__init__()

    def update(self, data, ids):
        return data


class SmoothingEMA(SmoothingBase):
    def __init__(self, weight) -> None:
        super().__init__()

        self.weight = weight

    def _update(self, cur_state, x):
        if cur_state is not None:
            y = cur_state * (1 - self.weight) + x * self.weight
        else:
            y = x

        return y, y

    def update(self, data, ids):
        new_state = {}
        out = np.zeros_like(data)

        for i, (id, x) in enumerate(zip(ids, data)):
            id = int(id)

            if id in self.state:
                y = self.state[id] * (1 - self.weight) + x * self.weight
            else:
                y = x

            new_state[id] = y
            out[i] = y

        self.state = new_state

        return out


class SmoothingSmoothDamp(SmoothingBase):
    def __init__(self, smooth_time) -> None:
        super().__init__()

        self.smooth_time = smooth_time

    def _new_state(self, x):
        vel = np.zeros_like(x)
        s = np.stack((x, vel))
        return s

    def _update_batch(self, last_state_data, cur_data):
        last_x, last_vel = last_state_data[:, 0], last_state_data[:, 1]
        new_x, new_vel = _smooth_damp(last_x, cur_data, last_vel, self.smooth_time)
        new_state_data = np.stack((new_x, new_vel), 1)
        return new_state_data, new_x

    """
    def _update(self, cur_state, x):
        if cur_state is not None:
            cur, cur_vel = cur_state
            y, vel = _smooth_damp(cur, x, cur_vel, self.smooth_time)
        else:
            y, vel = x, np.zeros_like(x)

        return y, (y, vel)
    """


class SmoothingConv(SmoothingBase):
    def __init__(self, kernel) -> None:
        super().__init__()

        if isinstance(kernel, int):
            kernel = np.full(kernel, 1/kernel)

        self.kernel = np.array(kernel).astype(np.float32)
        self.n = self.kernel.shape[0]

    def _new_state(self, x):
        s = np.repeat(x[None], self.n, 0)
        return s

    def _update_batch(self, last_state_data, cur_data):
        new_state_data = last_state_data  # n x k x d
        new_state_data[:, :-1] = new_state_data[:, 1:]  # roll
        new_state_data[:, -1] = cur_data  # insert new data
        new_x = np.einsum("nkd,k->nd", new_state_data, self.kernel)
        return new_state_data, new_x

    """
    def _update(self, cur_state, x):
        if cur_state is not None:
            vals = cur_state + [x]

            if len(vals) >= self.n:
                vals = vals[-self.n:]
                y = np.einsum("nd,nd->d", vals, self.kernel)
            else:
                y = x
        else:
            vals = [x]
            y = x

        return y, vals
    """
