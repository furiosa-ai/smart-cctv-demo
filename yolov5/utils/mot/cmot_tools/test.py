import os
import subprocess
import time
import numpy as np

from utils.cytest import assert_equal
from utils.mot.box_decode import box_decode as box_decode_py
from utils.mot.cmot_tools import box_decode
from utils.util import load_args


def test_box_decode():
    print("testing test_box_decode()")
    kwargs = load_args("box_decode_test_args")

    out_c = box_decode(**kwargs)
    out_py = box_decode_py(**kwargs)

    print(out_c[0] - out_py[0])

    assert np.allclose(out_py, out_c)
    print("done")

    t1 = time.time()
    for _ in range(100):
        box_decode(**kwargs)
    t2 = time.time()
    for _ in range(100):
        out_py = box_decode_py(**kwargs)
    t3 = time.time()

    print("C speed: {} | Python speed: {}ms".format(*[x * 1000 / 100 for x in[t2 - t1, t3 - t2]]))


def main():
    # sp = subprocess.Popen("source /Users/kevin/anaconda3/etc/profile.d/conda.sh && conda activate torch && ./build.sh", shell=True, executable="/bin/bash")
    # sp = subprocess.Popen("source /Users/kevin/anaconda3/etc/profile.d/conda.sh && conda run -n torch ./build.sh", shell=True, executable="/bin/bash")
    # sp.communicate()

    # assert sp.returncode == 0

    test_box_decode()


if __name__ == "__main__":
    main()
