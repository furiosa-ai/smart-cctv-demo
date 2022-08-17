import numpy as np
import time
import os

from multiprocessing import Process, Lock
from furiosa.runtime import session
from furiosa.tools.compiler.api import compile


def test_model_infer(model_file, device, lk=None, compile_enf=False):
    # test_model_file = "out/runs_train_bytetrack_mot20_5data_weights_best_512x512.enf"
    # test_model_file = "../onnx/exp/conv_test_random_quant_dfg.onnx"

    # furiosa_file = model_file.replace(".onnx", f".enf")

    if compile_enf:
        furiosa_file = model_file.replace(".onnx", f".enf")
        if not os.path.isfile(furiosa_file):
            compile(model_file, furiosa_file)
    else:
        furiosa_file = model_file

    # if not os.path.isfile(furiosa_file):
    #     compile(model_file, furiosa_file)

    print("Creating session...")
    with session.create(furiosa_file, device=device) as sess:
        print("Session created")
        inputs = [t for t in sess.inputs()]
        shapes = [i.shape for i in inputs]

        tensors = [np.zeros(shape, dtype=np.float32) for shape in shapes]
        
        print(f"Running first inference on {device}")
        outputs = sess.run(tensors)
        print(f"First inference completed on {device}")

        outputs_desc = [o.desc for o in outputs]
        print("Inputs:", inputs)
        print("Outputs:", outputs_desc)

        if lk is not None:
            lk.acquire()

        for _ in range(100):
            tensors = [np.random.rand(*shape).astype(np.float32) for shape in shapes]
            t1 = time.time()
            sess.run(tensors)
            print(f"{device} inference took {(time.time() - t1) * 1e3}ms")


def test_seq():
    # test_model_file = "out/runs_train_bytetrack_mot20_5data_weights_best_512x512_i8_dfg.onnx"
    # test_model_file = "out/runs_train_bytetrack_mot20_5data_weights_best_512x512.enf"
    # test_model_file = "out/test_model/yolo_quant.onnx"

    test_model_files = [
        "out/yolov5m_512_b1.onnx",
        "out/yolov5m_512_b2.onnx",
        # "out/yolov5m_512_b3.onnx",
        "out/yolov5m_512_b4.onnx",
    ]

    npus = [
        None
        # "npu1pe0",
        # "npu3pe1",
    ]


    for test_model_file in test_model_files:
        for npu in npus:
            test_model_infer(test_model_file, npu)

def test_multi_npu_by_sdk():
    test_model_file = "out/runs_train_bytetrack_mot20_5data_weights_best_512x512_b1_i8_dfg.onnx"
    npus = [
        "npu0pe0",
        "npu0pe1",
        "npu1pe0",
        "npu1pe1",
    ]
    npu = ','.join(npus)

    t = time.time()
    test_model_infer(test_model_file, npu)
    print('TOTAL:', time.time()-t)


def test_par():
    # test_model_file = "../onnx/exp/conv_test_random_quant_dfg.onnx"
    # test_model_file = "out/runs_train_bytetrack_mot20_5data_weights_best_512x512_i8_dfg.onnx"
    # test_model_file = "out/yolo_test.enf"
    # test_model_file = "out/yolov5m_512_b1.onnx"
    test_model_file = "out/runs_train_bytetrack_mot20_5data_weights_best_512x512_b1_i8_dfg.onnx"

    npus = [
        "npu0pe0",
        "npu0pe1",
        "npu1pe0",
        "npu1pe1",
    ]

    lks = [Lock() for npu in npus]

    procs = [Process(target=test_model_infer, args=(test_model_file, npu, lk)) for npu, lk in zip(npus, lks)]
    

    for lk in lks:
        lk.acquire()

    for proc in procs:
        proc.start()

    t = time.time()
    
    for lk in lks:
        lk.release()


    for proc in procs:
        proc.join()
    print(time.time()-t)


def main():
    # test_seq()
    test_par()
    #test_multi_npu_by_sdk()


if __name__ == "__main__":
    main()
