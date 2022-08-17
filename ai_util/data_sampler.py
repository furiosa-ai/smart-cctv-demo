from math import ceil


class DataSampler:
    def __init__(self, data, worker_idx, num_workers) -> None:
        self.data = data
        self.worker_idx = worker_idx
        self.num_workers = num_workers

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class VideoDataSampler(DataSampler):
    def __init__(self, data, worker_idx, num_workers) -> None:
        super().__init__(data, worker_idx, num_workers)

        data_count_per_worker = ceil(len(data) / max(self.num_workers, 1))

        self.cur_frame_idx = -1
        self.start_frame_idx = worker_idx * data_count_per_worker
        self.end_frame_idx = min((worker_idx + 1) * data_count_per_worker, len(self.data))

    def __iter__(self):
        if self.data.is_open():
            self.data.close()
        self.data.open()
        self.data.set_frame_idx(self.start_frame_idx)
        self.cur_frame_idx = -1
        return self

    def __next__(self):
        sample = self.data.get_next()
        self.cur_frame_idx = sample[0]

        if self.cur_frame_idx < self.end_frame_idx:
            assert sample is not None
            return sample
        else:
            raise StopIteration