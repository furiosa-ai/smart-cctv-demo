# https://github.com/jkjung-avt/tensorrt_demos/issues/43

import os


from .torch_to_onnx import torch_to_onnx
from .model_to_openvino import model_to_openvino


def torch_to_openvino(model, output_file, onnx_opset=None, check_onnx_model=False,
                      input_shape=None, input_names=("inputs",), output_names=("outputs",), fp16=False):
    output_file_onnx = os.path.splitext(output_file)[0] + ".onnx"

    # print("Converting to ONNX")
    torch_to_onnx(model, output_file_onnx, input_shape=input_shape, opset=onnx_opset, check_model=check_onnx_model,
                  input_names=input_names, output_names=output_names)

    print("Converting to ONNX")
    model_to_openvino(output_file_onnx, output_file, input_shape=input_shape, fp16=fp16)
