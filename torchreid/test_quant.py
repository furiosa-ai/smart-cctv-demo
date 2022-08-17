
import os
import numpy as np
import onnx
from torchvision.transforms import Compose
from PIL import Image
from ai_util.inference_framework import CalibrationDatasetImage
from torchreid.data.transforms import build_transforms
from tqdm import tqdm


def quantize_onnx_exp(onnx_file, out_file, quant_data):
    from furiosa.quantizer.frontend.onnx import optimize_model
    import furiosa.quantizer_experimental
    from furiosa.quantizer_experimental import CalibrationMethod, Calibrator

    model = onnx.load_model(onnx_file)
    model = optimize_model(model)
    print('optimized model')
    model = model.SerializeToString()

    calibrator = Calibrator(model, CalibrationMethod.PERCENTILE, percentage=99.99)

    for calibration_data in quant_data:
        calibrator.collect_data([calibration_data])


    ranges = calibrator.compute_range()
    graph = furiosa.quantizer_experimental.quantize(model, ranges)
    print("quantized model")
    graph = bytes(graph)

    with open(out_file, "wb") as f:
        f.write(graph)

    return graph
    # onnx.save_model(graph, out_file)

    # return out_file


def test_dfg(file):
    from furiosa.runtime import session

    with open(file, "rb") as f:
        graph = f.read()

    x = np.random.rand(1, 3, 256, 128).astype(np.float32)

    with session.create(graph) as sess:
         y = sess.run(x)
    print(y)


def main():
    transform = Compose([
        lambda x: Image.fromarray(x), 
        build_transforms(width=128, height=256)[1]
    ])

    batch_size = 1
    dataset = CalibrationDatasetImage("../data/market1501/Market-1501-v15.09.15/bounding_box_train/*", limit=10, needs_preproc=False, transform=transform)

    def _load_batch(i):
        ind = range(i * batch_size, (i + 1) * batch_size)
        x = [dataset[i] for i in ind]
        x = np.stack(x)  # add batch dimension

        return x

    num_batches = len(dataset) // batch_size
    assert num_batches > 0

    # data = (_load(i) for i in tqdm(range(len(dataset)), desc="Loading calibration data", total=len(dataset)))
    data = [_load_batch(i) for i in tqdm(range(num_batches), desc="Loading calibration data", total=num_batches)]

    if True:  #not os.path.exists("test.dfg"):
        quantize_onnx_exp(
            onnx_file="im_r50_softmax_256x128_amsgrad.onnx",
            out_file="test.dfg",
            quant_data=data
        )

    test_dfg("test.dfg")


if __name__ == "__main__":
    main()
