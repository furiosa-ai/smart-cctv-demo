import numpy as np


class BiList:
    def __init__(self) -> None:
        self.list = []
        self.rev_map = {}

    def append(self, x):
        self.rev_map[x] = len(self.list)
        self.list.append(x)

    def get_index_of(self, x):
        return self.rev_map[x]

    def to_array_(self):
        self.list = np.array(self.list)

    def __getitem__(self, idx):
        return self.list[idx]

    def __len__(self):
        return len(self.list)

    def delete_range(self, indices):
        lst = BiList()

        indices = set(indices)

        for i in range(len(self)):
            if i not in indices:
                lst.append(self.list[i])

        return lst
