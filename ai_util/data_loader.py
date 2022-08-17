
import numpy as np
import torch
from math import ceil
from threading import Thread
from queue import Queue as ThreadQueue
from multiprocessing import Process, Queue
import copy

from ai_util.shared_mem import SharedMemQueue


class DataLoader:
    def __init__(self, data, batch_size, num_worker, data_sampler) -> None:
        self.data = data
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.queue_size = batch_size * 2

        self.workers = None
        self.qu = None
        self.data_iter = None
        self.data_sampler = data_sampler

    def _start(self):
        assert self.workers is None

        # avoid modifying data
        # shared_data_supported = [False for e in dummy_data]

        if self.num_worker > 0:
            # qu = Queue(maxsize=self.queue_size)
            dummy_data = next(iter(copy.deepcopy(self.data)))
            shared_data_supported = [isinstance(e, (np.ndarray, torch.Tensor)) for e in dummy_data]
            qu = SharedMemQueue(maxsize=self.queue_size, dummy_data=dummy_data, use_shared_data=shared_data_supported, safe_data_access=True)
            workers = [Process(target=self.data_worker, args=(i, qu)) for i in range(self.num_worker)]
        else:
            qu = ThreadQueue(maxsize=self.queue_size)
            workers = [Thread(target=self.data_worker, args=(0, qu))]

        for w in workers:
            w.start()

        self.workers = workers
        self.qu = qu
        self.data_iter = iter(self.collect_data())

    def _end(self):
        self.qu = None
        self.data_iter = None

        for w in self.workers:
            w.join()
        self.workers = None

    def __iter__(self):
        self._start()
        return self

    def __next__(self):
        batch = []

        if self.data_iter is None:
            raise StopIteration

        try:
            for _ in range(self.batch_size):
                sample = next(self.data_iter)
                batch.append(sample)
        except StopIteration:
            self._end()
        
        if len(batch) > 0:
            return batch
        else:
            raise StopIteration

    def data_worker(self, idx, qu: Queue):
        # data = self.data if idx == 0 else copy.deepcopy(self.data)
        data = self.data  # will be copied to process
        # from ai_util.dataset import ImageDataset
        # data = ImageDataset("../data/test_face/tc1/tom_cruise_test.mp4", limit=None, frame_step=5)

        for sample in self.data_sampler(data, idx, max(self.num_worker, 1)):
            assert sample is not None
            qu.put(sample)

        qu.put(None)

    def collect_data(self):
        finished_worker_count = 0

        while finished_worker_count < max(self.num_worker, 1):
            res = self.qu.get()

            if res is None:
                finished_worker_count += 1
            else:
                yield res


def _test():
    from tqdm import tqdm
    from ai_util.dataset import ImageDataset
    from ai_util.data_sampler import VideoDataSampler

    frame_step = 1
    data = ImageDataset("../data/test_face/tc1/tom_cruise_test.mp4", limit=None, frame_step=frame_step)
    expected_keys = list(range(0, len(data), frame_step))

    loader = DataLoader(data, 4, 2, VideoDataSampler)

    all_keys = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="Loading data")):
        keys, imgs = zip(*batch)
        # print(batch_idx, keys)
        all_keys += keys

    all_keys = sorted(all_keys)

    for expected_key, res_key in zip(expected_keys, all_keys):
        assert expected_key == res_key
    assert len(expected_keys) == len(all_keys)

    # all_keys = sorted(all_keys)
    # print(all_keys)
    # print(len(all_keys))

if __name__ == "__main__":
    _test()
