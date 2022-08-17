from contextlib import ExitStack
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Lock
from queue import Queue as ThreadQueue


class DynamicQueueInstance:
    def __init__(self, dummy_data=None, use_shared_data=None, use_threading=False, safe_data_access=True) -> None:
        self.dummy_data = dummy_data
        self.use_shared_data = use_shared_data
        self.use_threading = use_threading
        self.safe_data_access = safe_data_access

    def __call__(self, *args, **kwargs):
        if self.use_threading:
            return ThreadQueue(*args, **kwargs)
        elif self.dummy_data is None:
            return Queue(*args, **kwargs)
        else:
            return SharedMemQueue(*args, **kwargs, dummy_data=self.dummy_data, use_shared_data=self.use_shared_data, safe_data_access=self.safe_data_access)


class SharedMemQueue:
    def __init__(self, maxsize, dummy_data, use_shared_data, safe_data_access, *args, **kwargs) -> None:
        assert self._is_iterable(dummy_data)
        assert len(dummy_data) == len(use_shared_data)

        self.shm_objs = []

        for i, d in enumerate(dummy_data):
            a = None
            if isinstance(d, np.ndarray):
                if use_shared_data[i]:
                    a = SharedObjectPool(SharedNumpyArray, maxsize, kwargs=dict(init_arr=d, safe_data_access=safe_data_access))
            else:
                assert not use_shared_data[i], "Unsupported for SM"

            if a is None:
                a = UnsharedObjectPool()

            self.shm_objs.append(a)

        self.qu = Queue(maxsize, *args, **kwargs)

        if self._isnamedtupleinstance(dummy_data):
            nt = dummy_data._make([None] * len(dummy_data))
        else:
            nt = None

        self.nt = nt

    def _is_iterable(self, o):
        return isinstance(o, (tuple, list)) or self._isnamedtupleinstance(o)

    def _isnamedtupleinstance(self, x):
        t = type(x)
        b = t.__bases__
        if len(b) != 1 or b[0] != tuple: return False
        f = getattr(t, '_fields', None)
        if not isinstance(f, tuple): return False
        return all(type(n)==str for n in f)

    def get(self, *args, **kwargs):
        qu_data = self.qu.get(*args, **kwargs)

        if qu_data is None:
            return None

        # return self._get_data(self.shm_objs, qu_data)
        data = []

        for d, shm_obj in zip(qu_data, self.shm_objs):
            shm_idx = d
            data.append(shm_obj.get_data(shm_idx))

        if self.nt is not None:
            data = self.nt._make(data)

        return data

    def put(self, data, *args, **kwargs):
        if data is None:
            self.qu.put(None, *args, **kwargs)
        else:
            # qu_data = []
            # self._put_data(data, self.shm_objs, qu_data)
            # self.qu.put(qu_data, *args, **kwargs)
            # return

            qu_data = []

            for d, shm_obj in zip(data, self.shm_objs):
                shm_idx = shm_obj.set_data(d)
                qu_data.append(shm_idx)

            self.qu.put(qu_data, *args, **kwargs)

        """
        # rec funcs, too slow
        def _put_data(self, data, shm_objs, qu_data):
            if self._is_iterable(shm_objs):
                return [self._put_data(d, o, qu_data) for d, o in zip(data, shm_objs)]
            else:
                if shm_objs is None:
                    qu_data.append(data)
                else:
                    shm_objs.set_data(data)

        def _get_data(self, shm_objs, qu_data):
            if self._is_iterable(shm_objs):
                return [self._get_data(o, qu_data) for o in shm_objs]
            else:
                if shm_objs is None:
                    return qu_data.pop(0)
                else:
                    return shm_objs.get_data()
        """


class SharedObject:
    def __init__(self, init_data) -> None:
        self.data = None

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data


class UnsharedObjectPool:
    def __init__(self) -> None:
        pass

    def set_data(self, data):
        return data

    def get_data(self, data):
        return data


class SharedObjectPool:
    def __init__(self, shared_obj_cls, n, args=(), kwargs=None) -> None:
        if kwargs is None:
            kwargs = {}

        self.shm_objs = [shared_obj_cls(*args, **kwargs) for _ in range(n)]

        if n > 1:
            free_objs = Queue(n)
            for i in range(n):
                free_objs.put(i)
        else:
            free_objs = None

        self.free_objs = free_objs

    def set_data(self, data):
        i = self.free_objs.get() if self.free_objs is not None else 0
        self.shm_objs[i].set_data(data)
        return i

    def get_data(self, i):
        d = self.shm_objs[i].get_data()
        if self.free_objs is not None:
            self.free_objs.put(i, block=False)
        return d


class SharedNumpyArray(SharedObject):
    def __init__(self, init_arr, safe_data_access) -> None:
        super().__init__(init_arr)
        ctype = np.ctypeslib.as_ctypes_type(init_arr.dtype)
        self.shm_arr = mp.Array(ctype, init_arr.size, lock=False)
        self.np_arr = np.frombuffer(self.shm_arr, dtype=init_arr.dtype).reshape(init_arr.shape)

        self.lk = Lock() if safe_data_access else None

        print(f"Created SharedNumpyArray with shape {init_arr.shape}")

    def __getstate__(self):
        return (self.shm_arr, self.np_arr.shape, self.lk)

    def __setstate__(self, state):
        self.shm_arr, shape, self.lk = state
        self.np_arr = np.ctypeslib.as_array(self.shm_arr).reshape(shape)

    def set_data(self, data):
        with ExitStack() as stack:
            if self.lk is not None:
                stack.enter_context(self.lk)

            self.np_arr[:] = data

    def get_data(self):
        with ExitStack() as stack:
            if self.lk is not None:
                stack.enter_context(self.lk)

            arr = self.np_arr

            if self.lk is not None:
                arr = arr.copy()

            return arr
