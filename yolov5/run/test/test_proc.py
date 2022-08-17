
from multiprocessing import Process, Queue, Lock, Barrier
import multiprocessing
import platform


def func(qu):
    pass


class QueueWrapper:
    def __init__(self) -> None:
        self.qu = Queue()

def main():
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    qu = QueueWrapper()

    proc = Process(target=func, args=(qu,))
    proc.start()
    proc.join()


if __name__ == "__main__":
    main()
