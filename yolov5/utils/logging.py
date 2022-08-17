import functools
import os
import json
import glob
import shutil
import threading
from typing import Callable


from utils.util import CustomProcess

import time


_tracer = None

def _create_tracer(*args, **kwargs):
    global _tracer
    # on ubuntu tracer will somehow be copied to new processes
    # assert _tracer is None, "Tracer already exists"
    _tracer = Tracer(*args, **kwargs)


def _end_tracer():
    if _tracer is not None:
        _tracer.exit()


def _get_tracer():
    return _tracer


class Tracer:
    log_dir = "log"

    def __init__(self, is_main=False, proc_name=None) -> None:
        if is_main and proc_name is None:
            proc_name = "Main"

        self.tracer = []
        self.proc_name = proc_name
        self.is_main = is_main

        if is_main:
            shutil.rmtree(self.log_dir, ignore_errors=True)
            os.mkdir(self.log_dir)

    def add_raw(self, raw):
        # print(f"add raw {raw}")
        self.tracer.append(raw)

    def getts(self):
        # microsec
        return time.time() * 1e6
        # return time.perf_counter_ns() // 1000
    
    def exit(self):
        pname = os.getpid() if self.proc_name is None else self.proc_name
        pid = os.getpid()
        # tid = threading.get_native_id()

        # pname = psutil.Process(os.getpid()).name()

        thread_ids = {}

        def _get_thread_id(tid):
            if tid not in thread_ids:
                thread_ids[tid] = len(thread_ids)
            return thread_ids[tid]

        out = [
            {
                "pid": pid,
                "args": {"name": pname},
                **raw
            } for raw in self.tracer
        ]

        for o in out:
            o["tid"] = _get_thread_id(o["tid"])

        with open(f"{self.log_dir}/{os.getpid()}.json", "w") as f:
            json.dump(out, f)

        if self.is_main:
            log_files = glob.glob(f"{self.log_dir}/*.json")

            log_data = []
            for log_file in log_files:
                with open(log_file, "r") as f:
                    log_data += json.load(f)

            with open("result.json", "w") as f:
                json.dump(log_data, f)


class ProcessLogEventListener:
    def __init__(self) -> None:
        pass

    def on_start(self, proc):
        # print("proc started")
        _create_tracer(proc_name=proc.name)

    def on_end(self):
        pass
        _end_tracer()
        # print("proc ended")


class Logger:
    def __init__(self) -> None:
        self.enable = False
        self.filter = None

    def start(self):
        if not os.environ.get("USE_CUSTOM_PYTHON_TRACER", False):
            return

        self._set_enable(True)

    def _set_enable(self, enable):
        self.enable = enable
        _create_tracer(is_main=True)

        if enable:
            CustomProcess.event_listener = ProcessLogEventListener()

    def log_required(self, func: Callable):
        return self.enable and (self.filter is None or func.__code__.co_name in self.filter)

    def exit(self):
        if self.enable:
            _end_tracer()


logger = Logger()


def log_func(func: Callable) -> Callable:
    if not os.environ.get("USE_CUSTOM_PYTHON_TRACER", False):
        return func

    # if not isinstance(CustomProcess.event_listener, ProcessLogEventListener):
    #     return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracer = _get_tracer()
        if tracer is not None:
            # This should always be true
            start = tracer.getts()
            ret = func(*args, **kwargs)
            dur = tracer.getts() - start
            code = func.__code__
            tid = threading.get_ident()
            raw_data = {
                "ph": "X",
                "name": f"{code.co_name} ({code.co_filename}:{code.co_firstlineno})",
                "ts": start,
                "dur": dur,
                "cat": "FEE",
                "tid": tid
            }
            tracer.add_raw(raw_data)
        else:  # pragma: no cover
            raise Exception("This should not be possible")
        return ret

    return wrapper


# def log_func(func: Callable) -> Callable:
#     return log_sparse(func) if logger.log_required(func) else func
