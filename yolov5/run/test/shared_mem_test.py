from collections import namedtuple
from multiprocessing import shared_memory
from typing import Iterable
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
import ctypes

from utils.shared_mem import DynamicQueueInstance


"""
def to_shared_array(arr, ctype):
    shared_array = mp.Array(ctype, arr.size, lock=False)
    temp = np.frombuffer(shared_array, dtype=arr.dtype)
    temp[:] = arr.flatten(order='C')
    return shared_array


def to_numpy_array(shared_array, shape):
    '''Create a numpy array backed by a shared memory Array.'''
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)
"""


def to_shared_array(data):
    d_shape = data.shape
    d_type = np.int64
    d_size = np.dtype(d_type).itemsize * np.prod(d_shape)

    shm = shared_memory.SharedMemory(create=True, size=d_size)
    shm_name = shm.name
    # numpy array on shared memory buffer
    a = np.ndarray(shape=d_shape, dtype=d_type, buffer=shm.buf)
    # copy data into shared memory ndarray once

    return shm, a, (shm_name, d_shape, d_type)


def to_numpy_array(shared_info):
    shm_name, d_shape, d_type = shared_info
    ex_shm = shared_memory.SharedMemory(name=shm_name)
    # numpy array b uses the same memory buffer as a
    b = np.ndarray(shape=d_shape, dtype=d_type, buffer=ex_shm.buf)
    # changes in b will be reflected in a and vice versa...

    return ex_shm, b


def worker_shared(in_qu, out_qu):
    while True:
        shared_info = in_qu.get()

        if shared_info is None:
            break

        shm_name, d_shape, d_type = shared_info
        ex_shm = shared_memory.SharedMemory(name=shm_name)
        # numpy array b uses the same memory buffer as a
        b = np.ndarray(shape=d_shape, dtype=d_type, buffer=ex_shm.buf)

        t1 = time.time()

        ex_shm.close()


        out_qu.put(t1)


QueueInput = namedtuple("QueueInput", ("label", "arr"))

def worker_non_shared(in_qu, out_qu):
    while True:
        data = in_qu.get()

        if data is None:
            break

        label, a = data.label, data.arr
        t1 = time.time()

        print(label, a)

        out_qu.put(t1)


def ns_test():
    in1 = Queue(1)
    ou1 = Queue(1)

    data = np.zeros((1080, 1920, 3), np.uint8)
    # data = np.zeros((720, 1280, 3), np.uint8)
    # data = np.zeros((480, 640, 3), np.uint8)
    # data = 1

    p1 = Process(target=worker_non_shared, args=(in1,ou1))
    p1.start()

    time.sleep(3)

    for _ in range(10):
        t1 = time.time()
        in1.put(data)
        t2 = ou1.get()

        print(f"Took {(t2 - t1)*1000}ms")

    in1.put(None)
    p1.join()


def shared_test2():
    data = np.ones((1080, 1920, 3), np.uint8)

    # InputQueue = DynamicQueueInstance(QueueInput("", data), [False, True])
    InputQueue = DynamicQueueInstance()

    in1 = InputQueue(1)
    ou1 = Queue(1)

    # data = np.zeros((720, 1280, 3), np.uint8)
    # data = np.zeros((480, 640, 3), np.uint8)
    # data = 1



    p1 = Process(target=worker_non_shared, args=(in1,ou1))
    p1.start()

    time.sleep(3)

    for _ in range(10):
        t1 = time.time()
        in1.put(QueueInput("my_arr", data))
        t2 = ou1.get()

        print(f"Took {(t2 - t1)*1000}ms")

    in1.put(None)
    p1.join()


def shared_test():
    in1 = Queue(1)
    ou1 = Queue(1)

    data = np.zeros((1080, 1920, 3), np.uint8)

    p1 = Process(target=worker_shared, args=(in1,ou1))
    p1.start()

    time.sleep(3)

    shm, a, data_shared = to_shared_array(data)

    for _ in range(10):
        t1 = time.time()
        a[:] = data[:]
        in1.put(data_shared)
        t2 = ou1.get()


        print(f"Took {(t2 - t1)*1000}ms")

    shm.close()
    shm.unlink()

    in1.put(None)
    p1.join()


"""
class SharedMemQueue:
    def __init__(self, qu_types, *args, **kwargs) -> None:
        self.qu = Queue(*args, **kwargs)
        self.is_sm = [t == np.ndarray for t in qu_types]

        self.sm = None

    def _np_to_sm(self, data):
        d_shape = data.shape
        d_type = np.int64
        d_size = np.dtype(d_type).itemsize * np.prod(d_shape)

        shm = shared_memory.SharedMemory(create=True, size=d_size)
        shm_name = shm.name
        # numpy array on shared memory buffer
        a = np.ndarray(shape=d_shape, dtype=d_type, buffer=shm.buf)
        # copy data into shared memory ndarray once

        return shm, a, (shm_name, d_shape, d_type)

    def init_from_dummy_data(self, data):
        assert self.sm is None
        assert isinstance(data, (tuple, list))


class SharedNumpyArray:
    def __init__(self, dummy_data) -> None:
        self.d_shape = dummy_data.shape
        self.d_type = np.int64
        self.d_size = np.dtype(self.d_type).itemsize * np.prod(self.d_shape)

        self.shm = shared_memory.SharedMemory(create=True, size=self.d_size)
        self.shm_name = self.shm.name
        # numpy array on shared memory buffer
        self.np_a = np.ndarray(shape=self.d_shape, dtype=self.d_type, buffer=self.shm.buf)

        self.ex_shm = None
        self.is_parent = True

    def __getstate__(self):
        return (self.shm_name, self.d_shape, self.d_type)

    def __setstate__(self, state):
        pass

    def set_data(self, data):
        self.np_a[:] = data

    def get_data(self):
        return SharedNumpyArrayView(self)

    def close(self):
        self.shm.close()

        if self.is_parent:
            self.shm.unlink()
        
        self.shm = None


class SharedNumpyArrayView:
    def __init__(self, sna) -> None:
        self.shm = None
        self.sna = sna

    def lock(self):
        self.shm = shared_memory.SharedMemory(name=self.sna.shm_name)
        b = np.ndarray(shape=self.sna.d_shape, dtype=self.sna.d_type, buffer=self.shm.buf)
        return b

    def unlock(self):
        self.shm.close()
        self.shm = None

    def __enter__(self):
        return self.lock()

    def __exit__(self, type, value, traceback):
        self.unlock()


class SharedMemQueueParent:
    def __init__(self) -> None:
        pass
"""


def main():
    # ns_test()
    shared_test2()


if __name__ == "__main__":
    main()
