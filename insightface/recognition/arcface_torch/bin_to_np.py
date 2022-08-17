from pathlib import Path
import numpy as np
import torch



def bin_to_np(path):
    from eval import verification
    out_path = Path(path).with_suffix(".npz")

    data_list, issame_list = verification.load_bin(path, (112, 112))
    data_list = np.stack([t.numpy() for t in data_list])

    data_list = data_list.astype(np.uint8)
    issame_list = np.array(issame_list)

    np.savez(file=out_path, data_list=data_list, issame_list=issame_list)


def load_np(path, image_size, limit=None):
    assert image_size == (112, 112)
    path = Path(path).with_suffix(".npz")
    data = np.load(path)
    data_list, issame_list = data["data_list"], data["issame_list"]

    if limit is not None:
        # take from front and back
        assert limit % 2 == 0
        l2 = limit // 2
        bins = bins[:l2*2] + bins[-l2*2:]
        issame_list = issame_list[:l2] + issame_list[-l2:]
        assert len(bins) == 2 * len(issame_list)

    print(f"Loaded {path}")
    data_list = data_list.astype(np.float32)
    data_list = torch.from_numpy(data_list)

    return data_list, issame_list


def main():
    paths = [
        'datasets/ms1m-retinaface-t1/lfw.bin',
        'datasets/ms1m-retinaface-t1/cfp_fp.bin',
        'datasets/ms1m-retinaface-t1/agedb_30.bin',
    ]

    for path in paths:
        bin_to_np(path)


if __name__ == "__main__":
    main()
