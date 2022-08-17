from collections import namedtuple
import time
import os
import pickle
import multiprocessing as mp
import threading
import queue

from utils.shared_mem import DynamicQueueInstance


def create_namedtuple_on_main(typename, field_names):
    import __main__
    namedtupleClass = namedtuple(typename, field_names)
    setattr(__main__, namedtupleClass.__name__, namedtupleClass)
    namedtupleClass.__module__ = "__main__"
    return namedtupleClass


class PerfMeasure:
    def __init__(self, name, rec=None) -> None:
        self.name = name
        self.t1 = None
        self.rec = rec

    def __enter__(self):
        self.t1 = time.time()
        return self

    def __exit__(self, type, value, tb):
        delta = time.time() - self.t1
        # print(f"{self.name} took {delta * 1000:.3f}ms")

        if self.rec is not None:
            self.rec.append(delta)


def dump_args(name, dic):
    path = "out/test_data"

    os.makedirs(path, exist_ok=True)

    file = os.path.join(path, name + ".pkl")

    with open(file, "wb") as f:
        pickle.dump(dic, f)


def load_args(name):
    path = "out/test_data"

    file = os.path.join(path, name + ".pkl")

    with open(file, "rb") as f:
        return pickle.load(f)


# from multiprocessing import Process, Lock, Queue
# from threading import Thread as Process, Lock; from queue import Queue

class CustomProcess(mp.Process):
    event_listener = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if CustomProcess.event_listener is not None:
            self.event_listener = CustomProcess.event_listener

    def run(self) -> None:
        self.on_start()
        super().run()
        self.on_end()

    def on_start(self):
        if self.event_listener is not None:
            CustomProcess.event_listener = self.event_listener
            self.event_listener.on_start(self)

    def on_end(self):
        if self.event_listener is not None:
            self.event_listener.on_end()


"""
MPImport = namedtuple("MPImport", ["Process", "Queue", "Lock", "Barrier"])
def mp_lib(multiproc):
    if multiproc == "process":
        # from multiprocessing import Process, Queue, Lock, Barrier
        out = MPImport(CustomProcess, mp.Queue, mp.Lock, mp.Barrier)
    elif multiproc == "thread":
        # from threading import Thread as Process, Lock, Barrier; from queue import Queue
        out = MPImport(threading.Thread, queue.Queue, threading.Lock, threading.Barrier)
    else:
        raise Exception(multiproc)

    
    return out
"""


class _MPLib:
    def __init__(self) -> None:
        self.multiproc = None
        self.Process = None
        self.Lock = None
        self.Barrier = None
        self.Queue = None

        # self._queue_instances = None

    def set_strategy(self, multiproc):
        assert multiproc in ("process", "thread")

        self.multiproc = multiproc

        if multiproc == "process":
            self.Process = CustomProcess
            self.Queue = mp.Queue
            self.Lock = mp.Lock
            self.Barrier = mp.Barrier
        elif multiproc == "thread":
            self.Process = threading.Thread
            self.Queue = queue.Queue
            self.Lock = threading.Lock
            self.Barrier = threading.Barrier
        else:
            raise Exception(multiproc)

    def create_dyn_queue(self, *args, **kwargs):
        qu = DynamicQueueInstance(*args, **kwargs, use_threading=self.multiproc == "thread")
        # self._queue_instances[name] = qu
        return qu

    """
    def NamedQueue(self, name):
        return self._queue_instances[name]
    """



MPLib = _MPLib()
