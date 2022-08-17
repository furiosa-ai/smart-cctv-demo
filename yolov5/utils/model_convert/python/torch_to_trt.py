# https://github.com/jkjung-avt/tensorrt_demos/issues/43

import os


from .torch_to_onnx import torch_to_onnx
from .onnx_to_trt import onnx_to_trt


def torch_to_trt(model, output_file, batch_stream=None, precision=None, onnx_opset=None, check_onnx_model=False,
                 input_shape=None, input_names=("inputs",), output_names=("outputs",)):
    output_file_onnx = os.path.splitext(output_file)[0] + ".onnx"

    # print("Converting to ONNX")
    torch_to_onnx(model, output_file_onnx, input_shape=input_shape, opset=onnx_opset, check_model=check_onnx_model,
                  input_names=input_names, output_names=output_names)

    print("Converting to TensorRT")
    onnx_to_trt(output_file_onnx, output_file, batch_stream=batch_stream, precision=precision, input_shape=input_shape)
