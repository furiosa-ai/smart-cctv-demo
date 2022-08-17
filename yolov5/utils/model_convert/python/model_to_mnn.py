import os
import tempfile
import json


def _get_onnx_input_shape(model_path):
    import onnx

    model = onnx.load(model_path)
    assert len(model.graph.input)

    shape = []

    for input in model.graph.input:
        print(input.name, end=": ")
        # get type of input tensor
        tensor_type = input.type.tensor_type
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    shape.append(d.dim_value)
                    print(d.dim_value, end=", ")  # known dimension
                elif (d.HasField("dim_param")):
                    shape.append(d.dim_param)
                    print(d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    shape.append(-1)
                    print("?", end=", ")  # unknown dimension with no name
        else:
            print("unknown rank", end="")

    return shape


def model_to_mnn(model_type, input_file, output_file, quantize_config=None):
    print("Converting to MNN")

    mnn_root = "/home/kevin/MNN/build/release_x86"

    cmd = f"bash -c '\n" \
          f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{mnn_root}/tools/converter/\n" \
          f"{mnn_root}/MNNConvert -f {model_type.upper()} --modelFile {input_file} --MNNModel {output_file} --bizCode biz\n"

    if quantize_config is not None:
        preproc_config_file = f"{mnn_root}/preprocessConfig.json"
        _, _, h, w = _get_onnx_input_shape(input_file)

        config = {
            "format": "RGB",
            "width": w,
            "height": h,
            "path": "/home/kevin/Documents/projects/dod/res/data/BDD100K/val/"
        }

        config.update(quantize_config)

        cmd += "\n"

        with open(preproc_config_file, "w") as f:
            json.dump(config, f)

        cmd += f"{mnn_root}/quantized.out {output_file} {output_file} {preproc_config_file}"

    cmd += f"'"

    print(cmd)
    os.system(cmd)
