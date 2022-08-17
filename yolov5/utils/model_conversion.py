from collections import namedtuple
import tempfile
from types import SimpleNamespace

from numpy.lib.function_base import median
from torch.nn.modules.module import Module
from matplotlib.pyplot import xkcd
import onnx
import numpy as np
import argparse
import os
import onnxruntime
from collections import namedtuple

import yaml

from utils import model_convert

import torch
from torch import nn
import glob

from utils.util import PerfMeasure


def _load_model(model):
    assert callable(model)
    m = model if isinstance(model, nn.Module) else model()
    assert isinstance(m, nn.Module)
    return m


def check_model_quantize(onnx_file, outpath=None, mode=0):
    from furiosa.quantizer.frontend.onnx import post_training_quantization_with_random_calibration
    from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode

    model = onnx.load_model(onnx_file)
    quant_model = post_training_quantization_with_random_calibration(model, True, True, QuantizationMode.DFG, 1)

    if outpath is None:
        outpath = f"{os.path.splitext(onnx_file)[0]}_i8.onnx"
    onnx.save_model(quant_model, f'{outpath}')

    return True

"""
def quantize_onnx(onnx_file, out_file, quant_data, fake_quant=False):
    from furiosa.quantizer.frontend.onnx import optimize_model, quantize
    from furiosa.quantizer.frontend.onnx.calibrate import calibrate
    from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode

    model = onnx.load_model(onnx_file)
    optimized_model = optimize_model(model)
    print('optimized model')

    dynamic_ranges = calibrate(optimized_model, quant_data)

    # out_file = f'{os.path.splitext(onnx_file)[0]}_i8.onnx'

    if not fake_quant:
        print('i8 model')
        quant_model = quantize(optimized_model, True, True, QuantizationMode.DFG, dynamic_ranges)
        onnx.save_model(quant_model, out_file)
    else:
        print('fake quant model')
        quant_model = quantize(optimized_model, True, True, QuantizationMode.FAKE, dynamic_ranges)
        onnx.save_model(quant_model, out_file)

    return out_file
"""


def quantize_onnx(onnx_file, out_file, quant_data, calib_mode=None):
    from furiosa.quantizer.frontend.onnx import optimize_model
    from furiosa.quantizer.frontend.onnx.calibrate import calibrate
    from furiosa.quantizer.frontend.onnx.quantizer.utils import QuantizationMode
    from furiosa.quantizer.frontend.onnx.quantizer import quantizer

    if calib_mode is None:
        calib_mode = QuantizationMode.DFG

    model = onnx.load_model(onnx_file)
    optimized_model = optimize_model(model)
    print('optimized model')

    dynamic_ranges = calibrate(optimized_model, quant_data)

    quant_model = quantizer.FuriosaONNXQuantizer(
        optimized_model, True, True, calib_mode, dynamic_ranges
    ).quantize()

    onnx.save_model(quant_model, out_file)

    return out_file


class _OnnxInferModule(nn.Module):
    def __init__(self, onnx_file, input_names, output_names) -> None:
        super().__init__()

        if not isinstance(input_names, (tuple, list)):
            input_names = [input_names]

        self.input_names = input_names
        self.output_names = output_names
        self.ort_session = onnxruntime.InferenceSession(onnx_file)

        if output_names is not None:
            self.Output = namedtuple(f"{os.path.basename(os.path.splitext(onnx_file)[0])}_Output", self.output_names)
        else:
            self.Output = None

    def get_input_shapes(self):
        inputs = self.ort_session.get_inputs()
        shapes = [i.shape for i in inputs]
        return shapes

    def forward(self, *x):
        # input_dict = {k: x.numpy() for k, x in input_dict.items()}
        input_dict = {k: v.numpy() if not isinstance(v, np.ndarray) else v for k, v in zip(self.input_names, x)}
        # out = self.ort_session.run(self.output_names, input_dict)
        out = self.ort_session.run(None, input_dict)
        # out = [torch.tensor(o) for o in out]

        if self.Output:
            out = self.Output(*[torch.tensor(v) for v in out])
        return out


class CalibrationDataPipeline:
    def __init__(self, transforms, data_out_name=None, model_in_name=None, batch_size=1) -> None:
        self.transforms = transforms
        self.data_out_name = data_out_name
        self.model_in_name = model_in_name
        self.batch_size = batch_size

    def add_transform(self, transform):
        self.transforms.append(transform)

    def set_transform(self, idx, transform):
        self.transforms[idx] = transform

    def __len__(self):
        return len(self.transforms[0]) // self.batch_size

    def _load_item(self, key):
        x = key
        for transform in self.transforms:
            x = transform(x)

        if self.data_out_name is not None:
            x = x[self.data_out_name]

        return x

    def __getitem__(self, key):
        ind = range(key * self.batch_size, (key + 1) * self.batch_size)

        x = [self._load_item(i) for i in ind]
        
        if isinstance(x, torch.Tensor):
            x = torch.stack(x, 0)
        else:
            x = np.stack(x, 0)

        if self.model_in_name is not None:
            x = {self.model_in_name: x}
            
        return x


class OnnxInfer(nn.Module):
    def __init__(self, model, input_shape, output_names, onnx_file, quant=None, calib_data=None, test_data=None, use_cache=False, batch_size=1,
        **kwargs):
        super().__init__()

        def convert_to_onnx():
            if isinstance(model, str):
                assert model.endswith(".onnx")
                return model
            elif not (use_cache and os.path.exists(onnx_file)):
                m = _load_model(model)

                model_convert.torch_to_onnx(m, onnx_file, input_shape, output_names=output_names, runtime_test=False, batch_size=batch_size, **kwargs)

            return onnx_file

        if quant == "fake" or quant == "dfg":
            assert calib_data is not None
            onnx_quant_file = f'{os.path.splitext(onnx_file)[0]}_i8_{quant}.onnx'

            if not (use_cache and os.path.exists(onnx_quant_file)):
                quantize_onnx(convert_to_onnx(), onnx_quant_file, OnnxInfer._get_data_samples(calib_data))
            
            model_file = onnx_quant_file
        elif quant is None:
            model_file = convert_to_onnx()
        else:
            assert False

        # no dfg calib supported, use with FuriosaInfer
        self.model = _OnnxInferModule(model_file, "input", output_names) if quant != "dfg" else None
        self.model_file = model_file
        self.output_names = output_names

        # if quant == "fake" and test_data is not None:
        #     OnnxInferModule._cmp_model_output(model, self.model, test_data)

    def get_input_shapes(self):
        return self.model.get_input_shapes()

    def load_model(self, model):
        if isinstance(model, str):
            assert model.endswith(".onnx")
            return model
        elif not (self.use_cache and os.path.exists(self.onnx_file)):
            model_convert.torch_to_onnx(model,  
                self.onnx_file, self.input_shape, output_names=self.output_names, runtime_test=False, **self.conv_args)

        return self.onnx_file

    @staticmethod
    def _cmp_model_output(model_base, model_cmp, data):
        crit = RMSELoss()
        metrics = {
            "min": np.min,
            "max": np.max,
            "mean": np.mean,
            "median": np.median
        }
        data = OnnxInfer._get_data_samples(data)

        diffs = []
        for x in data:
            out_base, out_cmp = [model(x) for model in [model_base, model_cmp]]
            diff = crit(out_base, out_cmp).numpy()
            diffs.append(diff)

        metric_vals = {k: m(diffs) for k, m in metrics}
        print(metric_vals)

    @staticmethod
    def _get_data_samples(data):
        if isinstance(data, CalibrationDataPipeline):
            data = [data[i] for i in range(len(data))]
        elif callable(data):
            data = data()
        return data

    def forward(self, x):
        out = self.model(x)
        return out


"""
class FuriosaInfer:
    def __init__(self, model, output_names=None):
        assert os.path.isfile(model)

        from furiosa.runtime import session

        self.sess = session.create(str(model))

        self.Output = namedtuple("FuriosaOutput", output_names) if output_names is not None else None

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        
        assert isinstance(x, np.ndarray)

        outputs = self.sess.run(x)

        outputs = [outputs[i].numpy() for i in range(len(outputs))]

        outputs = [torch.tensor(o) for o in outputs]

        if self.Output is not None:
            outputs = self.Output(*outputs)

        return outputs

    def to(self, *args, **kwargs):
        return self # no need to move to device here

    def eval(self):
        return self # no train or eval mode

    def close(self):
        self.sess.close()

    def get_input_shapes(self):
        inputs = [t for t in self.sess.inputs()]
        shapes = [i.shape() for i in inputs]
        return shapes

    @staticmethod
    def from_pytorch_model(*args, **kwargs):
        onnx_infer = OnnxInfer(*args, **kwargs, quant="dfg")
        infer = FuriosaInfer(onnx_infer.model_file, onnx_infer.output_names)

        return infer
"""


class FuriosaInfer(nn.Module):
    num_inst = 0
    devices = None

    def __init__(self, model, output_names=None, device_name=None, input_type="f32", input_format="chw"):
        super().__init__()

        assert os.path.isfile(model)

        from furiosa.runtime import session

        if device_name is None:
            device = FuriosaInfer.get_next_device()
        else:
            assert isinstance(device_name, int)
            device = FuriosaInfer.get_device(device_name)

        if not (input_type == "f32" and input_format == "chw"):
            assert input_type in ("f32", "i8")
            assert input_format in ("chw", "hwc")

            assert not (input_type == "f32" and input_format != "chw")
            assert input_type == "i8", "Nothing to do"

            compile_config = {
                "without_quantize": {
                    "parameters": [
                        {
                            "input_min": 0.0, "input_max": 1.0, 
                            "permute": [0, 2, 3, 1] if input_format == "hwc" else [0, 1, 2, 3]
                        }
                    ]
                },
                #"use_pdb": True,
                #"ignore_default_pdb": False
            }

            # TODO: temporary hack
            _, compile_config_file = tempfile.mkstemp(suffix = '.yaml')

            with open(compile_config_file, "w") as f:
                yaml.dump(compile_config, f)

            os.environ["NPU_COMPILER_CONFIG_PATH"] = compile_config_file

        self.sess = session.create(str(model), device=device)

        self.Output = namedtuple("FuriosaOutput", output_names) if output_names is not None else None

        FuriosaInfer.num_inst += 1

    @staticmethod
    def get_device(idx):
        if FuriosaInfer.devices is None:
            if 'NPU_DEVNAME' in os.environ and '0-1' in os.environ['NPU_DEVNAME']:
                # use fusion device
                    FuriosaInfer.devices = sorted(glob.glob("/dev/npu*pe0-1"))
                    FuriosaInfer.devices = [os.path.basename(d) for d in FuriosaInfer.devices]
            else:
                FuriosaInfer.devices = sorted(glob.glob("/dev/npu*pe?"))
                FuriosaInfer.devices = [os.path.basename(d) for d in FuriosaInfer.devices]
        else:
            FuriosaInfer.devices = sorted(glob.glob("/dev/npu*pe?"))
            FuriosaInfer.devices = [os.path.basename(d) for d in FuriosaInfer.devices]

        return FuriosaInfer.devices[idx]

    @staticmethod
    def get_next_device():
        return FuriosaInfer.get_device(FuriosaInfer.num_inst)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        assert isinstance(x, np.ndarray)

        with PerfMeasure("sess.run"):
            outputs = self.sess.run(x)

        outputs = [outputs[i].numpy() for i in range(len(outputs))]

        outputs = [torch.from_numpy(o) for o in outputs]

        if self.Output is not None:
            outputs = self.Output(*outputs)

        return outputs

    def to(self, *args, **kwargs):
        return self # no need to move to device here

    def eval(self):
        return self # no train or eval mode

    def close(self):
        self.sess.close()
        FuriosaInfer.num_inst -= 1

    @staticmethod
    def from_pytorch_model(furiosa_model_file=None, use_cache=False, device_name=None, input_type="f32", input_format="chw", *args, **kwargs):
        from furiosa.tools.compiler.api import compile

        # enf loading not working well
        furiosa_model_file = None

        # device = FuriosaInfer.get_next_device()

        # if furiosa_model_file is not None:
        #     name, ext = os.path.splitext(furiosa_model_file)
        #     furiosa_model_file = f"{name}_{device}{ext}"

        def _to_onnx():
            oi = OnnxInfer(use_cache=use_cache, *args, **kwargs, quant="dfg")
            return oi.model_file  #, oi.output_names

        if furiosa_model_file is not None:
            if use_cache and os.path.isfile(furiosa_model_file):
                pass
            else:
                onnx_file = _to_onnx()
                compile(onnx_file, furiosa_model_file)  # , target_npu=device
        else:
            furiosa_model_file = _to_onnx()

        output_names = None

        infer = FuriosaInfer(furiosa_model_file, output_names, device_name=device_name, input_type=input_type, input_format=input_format)

        return infer


def model_to_device(
    pytorch_model, 
    device, 
    input_shape,
    output_names,
    calib_data,
    model_path,
    model_name,
    input_type="f32",
    input_format="chw",
    batch_size=1,
    device_name=None,
    quant=None,
    use_cache=True):

    onnx_file = os.path.join(model_path, model_name) + f"_{input_shape[2]}x{input_shape[1]}_b{batch_size}.onnx"
    furiosa_model_file = onnx_file.replace(".onnx", ".enf")
    convert_params = vars(SimpleNamespace(
        input_shape=input_shape,
        output_names=output_names,
        calib_data=calib_data,
        onnx_file=onnx_file,
        # furiosa_model_file=furiosa_model_file,
        batch_size=batch_size,
        use_cache=use_cache
    ))

    if device in ("onnx", "onnx_i8"):
        convert_params["quant"] = (quant if quant is not None else "fake") if device == "onnx_i8" else None
        return OnnxInfer(
            pytorch_model, 
            **convert_params
        )
    elif device == "furiosa":
        convert_params["device_name"] = device_name
        convert_params["input_type"] = input_type
        convert_params["input_format"] = input_format
        return FuriosaInfer.from_pytorch_model(model=pytorch_model, **convert_params)
    else:
        return _load_model(pytorch_model).to(device)
