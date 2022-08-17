
import copy
from multiprocessing import Process, Queue, Lock
from pathlib import Path
import glob
import tempfile
from threading import Thread
from queue import Queue as ThreadQueue
import time
from typing import Dict, Iterable, Tuple, Optional
import cv2
import os
import torch
import random
import numpy as np
import onnx
import onnxruntime
from tqdm import tqdm

from ai_util.onnx_util import OnnxModelEdit, onnx_add_norm_layer


def _quantize_onnx(onnx_file, out_file, quant_data, calib_mode=None):
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


def _quantize_onnx_to_dfg_exp(onnx_model, out_file, calib_data, calib_mode):
    from furiosa.quantizer.frontend.onnx import optimize_model
    import furiosa.quantizer_experimental
    from furiosa.quantizer_experimental import CalibrationMethod, Calibrator

    if isinstance(onnx_model, (str, Path)):
        model = onnx.load_model(onnx_model)
    else:
        model = onnx_model

    model = optimize_model(model)
    print('Optimized model')
    model = model.SerializeToString()

    if calib_mode == "percentile":
        calibrator = Calibrator(model, CalibrationMethod.PERCENTILE, percentage=99.99)
    elif calib_mode == "entropy":
        calibrator = Calibrator(model, CalibrationMethod.ENTROPY)
    elif calib_mode == "minmax":
        calibrator = Calibrator(model, CalibrationMethod.MIN_MAX)
    else:
        raise Exception(calib_mode)

    for sample in tqdm(calib_data, "Computing ranges"):
        if isinstance(sample, dict):
            assert len(sample) == 1
            sample = list(sample.values())
        calibrator.collect_data(sample)

    ranges = calibrator.compute_range()
    print("Quantizing...")
    graph = furiosa.quantizer_experimental.quantize(model, ranges)
    print("Quantized model")
    graph = bytes(graph)

    with open(out_file, "wb") as f:
        f.write(graph)

    return graph


class PredictorBase:
    def __init__(self, name=None, quant_tag=None, calib_mode=None, input_format="chw", input_prec="f32", 
        device=None, skip_preproc=False, batch_size=1, pad_batch=False, norm_mean=None, norm_std=None) -> None:
        self.model = None
        self.name = name
        self.quant_tag = quant_tag
        self.calib_mode = calib_mode
        self.weights_loaded = False
        self.input_format = input_format
        self.input_prec = input_prec
        self._input_shapes = None
        self.device = None
        self.calib_mode = calib_mode
        self.skip_preproc = skip_preproc
        self.batch_size = batch_size
        self.pad_batch = pad_batch
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        assert (norm_mean is None and norm_std is None) or (norm_mean is not None and norm_std is not None)

        if device is not None:
            self.to(device)

    def get_dim_index(self, dim):
        return self.input_format.index(dim) + 1  # for batch size

    def get_model(self):
        if self.model is None:
            self.model = self.build_model()
        return self.model

    def get_calibration_dataset(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def preproc(self, x, input_format, input_prec):
        raise NotImplementedError
    
    def postproc(self, x, info):
        raise NotImplementedError

    def preproc_img(self, img, input_size, input_format, input_prec):
        if input_format is None:
            input_format = self.input_format

        if input_prec is None:
            input_prec = self.input_prec

        # ori_height, ori_width = img.shape[:2]

        # img_resized_list = []
        # for this_short_size in self.imgSizes:
            # calculate target height and width

        # to avoid rounding in network
        # target_width = self.round2nearest_multiple(target_width, self.padding_constant)
        # target_height = self.round2nearest_multiple(target_height, self.padding_constant)

        # resize images
        img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)

        if input_format == "chw":
            img = img.transpose(2, 0, 1)

        if input_prec == "f32":
            img = img.astype(np.float32) / 255

            if self.norm_mean is not None:
                mean, std = self.norm_mean, self.norm_std

                if isinstance(mean, np.ndarray):
                    mean = mean[:, None, None]
                    std = std[:, None, None]

                img = (img - mean) / std

        img = np.ascontiguousarray(img)

        return img

    def data_to_device(self, x):
        if self.device in ("cpu", "cuda"):
            return torch.from_numpy(x).to(self.device)
        else:
            return x

    def infer(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        assert x.shape[0] <= self.batch_size

        batch_pad = self.batch_size - x.shape[0]
        assert not (batch_pad > 0 and not self.pad_batch)

        if batch_pad > 0:
            x = np.concatenate((x, np.zeros((batch_pad, *x.shape[1:]), dtype=x.dtype)))

        x = self.data_to_device(x)

        with torch.no_grad():
            x = self.model(x)

        if batch_pad > 0:
            if isinstance(x, (tuple, list)):
                x = [t[:t.shape[0] - batch_pad] for t in x]
            else:
                x = x[:x.shape[0] - batch_pad]

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        return x

    def get_input_format(self):
        return self.input_format

    def get_input_prec(self):
        return self.input_prec

    def get_input_shapes(self, input_format="chw", input_prec="f32"):
        if self._input_shapes == None:
            dataset = self.get_calibration_dataset()
            sample = dataset[0]
            if isinstance(dataset, CalibrationDatasetBase) and dataset.needs_preproc:
                x, _ = self.preproc(sample, input_format=input_format, input_prec=input_prec)
            else:
                x = sample

            if isinstance(x, (tuple, list)):
                self._input_shapes = [(self.batch_size, *v.shape) for v in x]
            else:
                self._input_shapes = (self.batch_size, *x.shape)
        return self._input_shapes

    def get_sample_input(self):
        input_shapes = self.get_input_shapes()

        if isinstance(input_shapes[0], (tuple, list)):
            return [torch.zeros(s) for s in input_shapes]
        else:
            return torch.zeros(input_shapes)

    def collate_input(self, inputs):
        return np.stack(inputs)

    def decollate_output(self, outputs):
        return [*outputs]

    def to(self, device, out_file=None, no_load=False):
        if out_file is None:
            out_file = self.name

        assert out_file is not None

        device, *devid = device.split(":", 2)
        devid = devid[0] if len(devid) == 1 else None

        if device == "furiosa":
            self.model = FuriosaInferenceEngine(self, out_file, devid=devid)
        else:
            assert devid is None
            self.input_format = "chw"
            self.input_prec = "f32"
            if device == "onnx":
                self.model = OnnxInferenceEngine(self, out_file, no_load=no_load)
            elif device == "onnx_i8":
                self.model = OnnxInferenceEngineI8(self, out_file, no_load=no_load)
            elif device == "trt":
                self.model = TensorRTInferenceEngine(self, out_file, no_load=no_load)
            elif device in ("cpu", "cuda"):
                self.model = self.get_model().to(device)
            else:
                raise Exception(device)

        self.weights_loaded = True
        self.device = device
        return self

    def shared_inference(self, idx=None, device=None, num_inst=None):
        if idx is None:
            self.to(device, no_load=True)
            self.model = SharedInferenceEngine(self.model, num_inst)
        else:
            self.model = self.model.create_instance(idx)
        return self

    def create_shared_inference_engine(self, num_inst=None):
        return SharedInferenceEngine(self.model, num_inst)

    def connect_to_shared_inference(self, idx, in_qu, out_qus):
        self.model = SharedInferenceEngineInstance(idx, in_qu, out_qus)

    def start_inference(self):
        self.model.init_runtime()

    def end_inference(self):
        self.model.exit_runtime()

    def __call__(self, inputs, info=None):
        return self.predict(inputs, info=None)

    def predict(self, inputs, info=None):
        if not self.skip_preproc:
            assert info is None, "preproc info will be acquired from preproc func"
            is_single_input = not isinstance(inputs, (tuple, list))

            if is_single_input:
                inputs = [inputs]

            inputs, info = zip(*[self.preproc(x, input_format=self.input_format, input_prec=self.input_prec) for x in inputs])
            x = self.collate_input(inputs)
        else:
            x = inputs

        x = self.infer(x)
        x = self.postproc(x, info)

        # if is_single_input:
        #     x = self.decollate_output(x)[0]
        
        return x

    @torch.no_grad()
    def predict_all(self, input_iter):
        extra_all, preproc_params_all, batch_pad = [{} for _ in range(3)]

        def _input_iter():
            for idx, (inputs, extra) in enumerate(input_iter):
                if not self.skip_preproc:
                    inputs, preproc_params_all[idx] = zip(*[self.preproc(x, input_format=self.input_format, input_prec=self.input_prec) for x in inputs])
                    x = self.collate_input(inputs)
                else:
                    x = inputs

                # if isinstance(x, torch.Tensor):
                #     x = x.numpy()

                assert x.shape[0] <= self.batch_size

                batch_pad[idx] = self.batch_size - x.shape[0]
                assert not (batch_pad[idx] > 0 and not self.pad_batch)

                if batch_pad[idx] > 0:
                    x = np.concatenate((x, np.zeros((batch_pad[idx], *x.shape[1:]), dtype=x.dtype)))

                x = self.data_to_device(x)

                extra_all[idx] = extra
                yield x

        infer_iter = self.model.infer_all(_input_iter()) if not isinstance(self.model, torch.nn.Module) else ((i, self.model(x)) for i, x in enumerate(_input_iter()))

        for idx, x in infer_iter:
            if batch_pad[idx] > 0:
                if isinstance(x, (tuple, list)):
                    x = [t[:t.shape[0] - batch_pad[idx]] for t in x]
                else:
                    x = x[:x.shape[0] - batch_pad[idx]]

            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()

            res = self.postproc(x, preproc_params_all.pop(idx))
            extra = extra_all.pop(idx)

            yield res, extra

    def set_async_inference(self, enable):
        if not isinstance(self.model, torch.nn.Module):
            self.model.set_async_inference(enable)


class InferenceEngineBase:
    def infer(self, *x):
        raise NotImplementedError

    def infer_all(self, it):
        for idx, x in enumerate(it):
            out = self.infer(x)
            yield idx, out

    def set_async_inference(self, enable):
        pass

    def close(self):
        raise NotImplementedError   

    def __call__(self, *x):
        return self.infer(*x)

#     def __init__(self) -> None:
#         pass

#     def load_


class SharedInferenceEngineInstance():
    def __init__(self, idx, in_qu, out_qus) -> None:
        self.idx = idx
        self.in_qu = in_qu
        self.out_qus = out_qus

    def infer(self, x):
        self.in_qu.put((self.idx, x))
        print("put input")
        out = self.out_qus[self.idx].get()
        # out = [np.zeros(s, dtype=np.float32) for s in [(1, 48, 80, 80), (1, 48, 40, 40), (1, 48, 20, 20), ]]
        # out = copy.deepcopy(out)
        print("recv output")
        return out


class SharedInferenceEngine(InferenceEngineBase):
    def __init__(self, inference_engine, num_inst) -> None:
        super().__init__()

        self.inference_engine = inference_engine
        in_qu = Queue()
        out_qus = [Queue() for _ in range(num_inst)]
        self.idx = -1
        self.inst_count = 0

        # need to create after child procs are created
        self.worker = Process(target=self._run, args=(in_qu, out_qus))
        
        # self.worker = worker
        self.in_qu = in_qu
        self.out_qus = out_qus

    def init_runtime(self):
        self.worker.start()

    def exit_runtime(self):
        # TODO: exit proc
        self.in_qu.put(None)
        self.worker.join()
        self.worker = None

    def create_instance(self, idx):
        return SharedInferenceEngineInstance(idx, self.in_qu, self.out_qus)

    def _run(self, in_qu, out_qus):
        if not isinstance(self.inference_engine, torch.nn.Module):
            self.inference_engine.init_runtime()
        
        while True:
            assert self.idx == -1
            print("on recv input")
            res = in_qu.get()

            if res is None:
                break

            idx, inputs = res
            # idx, *inputs = self.in_qu.get()
            outputs = self.inference_engine(inputs)
            print("infered")
            out_qus[idx].put(outputs)
            # out_qus[idx].put(None)
            print("put output")

        
        print("finished infer")

        if not isinstance(self.inference_engine, torch.nn.Module):
            self.inference_engine.exit_runtime()


class OnnxInferenceEngine(InferenceEngineBase):
    def __init__(self, predictor: PredictorBase, out_file, no_load=False, opset_version=12, **kwargs) -> None:
        super().__init__()

        out_file = Path(out_file).with_suffix(".onnx")
        self.file = out_file
        self.runtime = None
        self.input_names = None

        if not out_file.exists():
            print(f"Converting to ONNX -> '{out_file}'")
            torch.onnx.export(predictor.get_model(), predictor.get_sample_input(), self.file, opset_version=opset_version, **kwargs)
            m = onnx.load(self.file)
            onnx.checker.check_model(m)
            onnx.shape_inference.infer_shapes(m)

        if not no_load:
            self.init_runtime()

    def init_runtime(self):
        assert self.runtime is None
        self.runtime = onnxruntime.InferenceSession(str(self.file))
        self.input_names = [i.name for i in self.runtime.get_inputs()]

    def exit_runtime(self):
        self.runtime = None

    def infer(self, x):
        if not isinstance(x, (tuple, list)):
            x = [x]

        input_dict = {k: v.numpy() if not isinstance(v, np.ndarray) else v for k, v in zip(self.input_names, x)}
        # out = self.ort_session.run(self.output_names, input_dict)
        out = self.runtime.run(None, input_dict)

        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]

        return out
    

class OnnxInferenceEngineI8(InferenceEngineBase):
    def __init__(self, predictor: PredictorBase, out_file, no_load=False, opset_version=12, **kwargs) -> None:
        super().__init__()

        out_file = Path(out_file)

        out_dir = out_file.parent
        out_filename = [out_file.name]

        if predictor.quant_tag is not None:
            out_filename.append(predictor.quant_tag)

        out_filename.append("i8")
        out_filename = "_".join(out_filename) + ".onnx"

        self.file = out_dir / out_filename
        self.input_format = predictor.get_input_format()
        self.input_prec = predictor.get_input_prec()
        self.input_names = None
        self.runtime = None
        self.batch_size = predictor.batch_size

        if not self.file.exists():
            onnx_infer = OnnxInferenceEngine(predictor, out_file, **kwargs)
            print(f"Quantizing -> '{self.file}'")
            self._quantize(predictor, onnx_infer)

        if not no_load:
            self.init_runtime()
    
    def _format_input(self, x):
        # for single input
        if not isinstance(x, (tuple, list)):
            x = [x]

        assert len(x) == len(self.input_names)

        x = {n: v for n, v in zip(self.input_names, x)}

        return x

    def _quantize(self, predictor, onnx_infer):
        self.input_names = onnx_infer.input_names
        dataset = predictor.get_calibration_dataset()

        def _load(i):
            x = dataset[i]
            if isinstance(dataset, CalibrationDatasetBase) and dataset.needs_preproc:
                x, _ = predictor.preproc(x, input_format="chw", input_prec="f32")
            return x

        def _load_batch(i):
            ind = range(i * self.batch_size, (i + 1) * self.batch_size)
            x = [_load(i) for i in ind]
            x = predictor.collate_input(x)  # add batch dimension
            x = self._format_input(x)

            return x

        num_batches = len(dataset) // self.batch_size
        assert num_batches > 0

        # data = (_load(i) for i in tqdm(range(len(dataset)), desc="Loading calibration data", total=len(dataset)))
        data = [_load_batch(i) for i in tqdm(range(num_batches), desc="Loading calibration data", total=num_batches)]
        _quantize_onnx(onnx_infer.file, self.file, data, calib_mode=predictor.calib_mode)

    def init_runtime(self):
        self.runtime = onnxruntime.InferenceSession(str(self.file))
        self.input_names = [i.name for i in self.runtime.get_inputs()]

    def close_runtime(self):
        pass

    def infer(self, x):
        if not isinstance(x, (tuple, list)):
            x = [x]

        input_dict = {k: v.numpy() if not isinstance(v, np.ndarray) else v for k, v in zip(self.input_names, x)}
        # out = self.ort_session.run(self.output_names, input_dict)
        out = self.runtime.run(None, input_dict)

        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]

        return out


class FuriosaInferenceEngine(InferenceEngineBase):
    def __init__(self, predictor: PredictorBase, out_file, devid=None, no_load=False, async_infer=False, **kwargs) -> None:
        super().__init__()

        self.use_exp_quantizer = predictor.calib_mode is not None
        self.batch_size = predictor.batch_size
        self.norm_mean = predictor.norm_mean
        self.norm_std = predictor.norm_std
        self.has_per_channel_scale = isinstance(self.norm_mean, Iterable)

        if not self.use_exp_quantizer:
            file = OnnxInferenceEngineI8(predictor, out_file, no_load=True).file
        else:
            file = self._gen_filename(out_file, predictor)
            if not file.is_file():
                onnx_infer = OnnxInferenceEngine(predictor, out_file, **kwargs)
                print(f"Quantizing ({predictor.calib_mode}) -> '{file}'")
                self._quantize(predictor, onnx_infer, file)

        self.file = file

        self.name = predictor.name
        self.input_format = predictor.get_input_format()
        self.input_prec = predictor.get_input_prec()
        self.input_names = None
        self.runtime = None
        self.devid = devid
        self.is_async = async_infer
        self.onnx_file_norm_suffix = "_dwnorm"

        if not no_load and not async_infer:
            self.init_runtime()

    @property
    def is_loaded(self):
        return self.runtime is not None

    def set_async_inference(self, enable):
        if enable != self.is_async:
            if self.is_loaded:
                self.close_runtime()

            self.is_async = enable

            if not enable:
                self.init_runtime()

    def _quantize(self, predictor, onnx_infer, file):
        onnx_model = onnx.load(onnx_infer.file)

        # per channel scaling
        if self.has_per_channel_scale:
            onnx_add_norm_layer(onnx_model, self.norm_mean, self.norm_std)
            # onnx.save_model(onnx_model, "log_normed_model.onnx")

        self.input_names = onnx_infer.input_names
        dataset = predictor.get_calibration_dataset()

        def _load(i):
            x = dataset[i]
            if isinstance(dataset, CalibrationDatasetBase) and dataset.needs_preproc:
                x, _ = predictor.preproc(x, input_format="chw", input_prec="f32")

            if self.has_per_channel_scale:
                # undo normalization which is handled in onnx itself now
                x = (x * self.norm_std[:, None, None]) + self.norm_mean[:, None, None]

            return x

        def _load_batch(i):
            ind = range(i * self.batch_size, (i + 1) * self.batch_size)
            x = [_load(i) for i in ind]
            x = predictor.collate_input(x)  # add batch dimension
            x = self._format_input(x)

            return x

        num_batches = len(dataset) // self.batch_size
        assert num_batches > 0

        data = [_load_batch(i) for i in tqdm(range(num_batches), desc="Loading calibration data", total=num_batches)]

        _quantize_onnx_to_dfg_exp(onnx_model, file, data, calib_mode=predictor.calib_mode)

    def _format_input(self, x):
        # for single input
        if not isinstance(x, (tuple, list)):
            x = [x]

        assert len(x) == len(self.input_names)

        x = {n: v for n, v in zip(self.input_names, x)}

        return x
    
    def _gen_filename(self, out_file, predictor):
        out_file = Path(out_file)

        out_dir = out_file.parent
        out_filename = [out_file.name]

        if predictor.quant_tag is not None:
            out_filename.append(predictor.quant_tag)

        out_filename.append("i8")
        out_filename = "_".join(out_filename) + (".onnx" if not self.use_exp_quantizer else ".dfg")

        file = out_dir / out_filename
        return file

    def init_runtime(self):
        from furiosa.runtime import session

        assert self.runtime is None

        if self.input_prec == "i8":
            assert self.input_prec == "i8"

            mean = self.norm_mean
            std = self.norm_std

            if self.has_per_channel_scale:
                # assert False, "no channel wise normalization yet"
                # assert self.file.stem.endswith(self.onnx_file_norm_suffix)
                mean, std = None, None

            input_min, input_max = 0.0, 1.0

            if mean is not None:
                input_min, input_max = [((x - mean) / std) for x in (input_min, input_max)]

            compile_config = {
                "without_quantize": {
                    "parameters": [
                        {
                            "input_min": input_min, "input_max": input_max, 
                            "permute": [0, 2, 3, 1] if self.input_format == "hwc" else [0, 1, 2, 3]
                        }
                    ]
                },
                #"use_pdb": True,
                #"ignore_default_pdb": False
            }
        else:
            assert self.input_format == "chw"
            compile_config = None

        print(f"Creating Furiosa session from file '{self.file}'")
        if not self.is_async:
            sess = session.create(str(self.file), device=self.devid, compile_config=compile_config)
            out_qu = None
        else:
            sess, out_qu = session.create_async(str(self.file), device=self.devid, compile_config=compile_config)

        self.runtime = sess
        self.out_qu = out_qu

        self.input_names = [i.name for i in self.runtime.inputs()]

    def close_runtime(self):
        assert self.runtime is not None

        if self.is_async:
            self.out_qu.close()

        self.runtime.close()
        self.runtime = None
        self.out_qu = None

    def infer_all(self, *args, **kwargs):
        if self.is_async:
            return self._infer_all_async(*args, **kwargs)
        else:
            return super().infer_all(*args, **kwargs)

    def _infer_all_async_worker(self, it, qu):
        for idx, x in enumerate(it):
            self.runtime.submit(x, idx)
            qu.put(True)

        qu.put(False)

    def _infer_all_async(self, it):
        from furiosa.runtime.errors import SessionClosed

        self.init_runtime()

        qu = ThreadQueue()
        worker = Thread(target=self._infer_all_async_worker, args=(it, qu))
        worker.start()

        while qu.get():
            idx, outputs = self.out_qu.recv()

            outputs = [outputs[i].numpy() for i in range(len(outputs))]
            yield idx, outputs

        worker.join()

        self.close_runtime()
        
    """
    def infer(self, *x):
        assert not self.is_async
        x = list(x)
        outputs = self.sess.run(x)

        outputs = [outputs[i].numpy() for i in range(len(outputs))]

        return outputs
    """

    def infer(self, x):
        # x = self._format_input(x)
        assert not self.is_async

        if isinstance(x, torch.Tensor):
            x = x.numpy()

        t1 = time.time()
        out = self.runtime.run(x)
        # print(f"[{self.name}] Infer took {(time.time() - t1) * 1000}ms")
        out = [o.numpy() for o in out]

        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]

        return out
        


class TensorRTInferenceEngine(InferenceEngineBase):
    def __init__(self, predictor: PredictorBase, out_file, **kwargs) -> None:
        from edge_ai.tensorrt.tensorrt_infer_dyn import TensorRTInferDyn
        from edge_ai.tensorrt.batch_stream import BatchStream
        super().__init__()
        
        out_file = Path(out_file)

        out_dir = out_file.parent
        out_filename = [out_file.name]
        out_filename = "_".join(out_filename) + ".trt"

        self.file = out_dir / out_filename
        self.input_names = None
        self.runtime = None

        if not self.file.exists():
            onnx_infer = OnnxInferenceEngine(predictor, out_file, **kwargs)
            input_shape = predictor.get_input_shapes()
            # assert len(input_shapes) == 1

            onnx_file = onnx_infer.file
            dataset = predictor.get_calibration_dataset()
            batch_stream = BatchStream(self.calib_data_wrapper(predictor, dataset))
        else:
            onnx_file = None
            input_shape = None
            batch_stream = None

        self.runtime = TensorRTInferDyn(str(self.file), str(onnx_file), batch_stream=batch_stream, precision="int8", input_shape=input_shape)

    def calib_data_wrapper(self, predictor, data):
        class _CalibData:
            def __init__(self):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                x = self.data[idx]
                if isinstance(self.data, CalibrationDatasetBase) and self.data.needs_preproc:
                    x, _ = predictor.preproc(x, input_format="chw", input_prec="f32")
                return x

        return _CalibData()

    def close_runtime(self):
        pass

    def infer(self, x):
        # x = self._format_input(x)
        x = np.ascontiguousarray(x)
        out = self.runtime(x)

        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]

        out = out.copy()  # otherwise memory may get overwritten
        return out


class InferenceEngineRemoteSocket:
    def __init__(self, target) -> None:
        pass

    def send_inputs(self, data):
        pass

    def recv_inputs(self):
        pass

    def send_outputs(self, data):
        pass

    def recv_outputs(self):
        pass

    def send_model(self, model):
        pass

    def recv_model(self):
        pass


class InferenceEngineRemoteClient:
    def __init__(self, engine) -> None:
        self.inputs = []
        self.pod = "kgalim-warboy-0"
        self.target_dir = "/root/projects/infer_server"
        self.server_script = "infer_server.sh"

        self.inputs_file_src = "inputs.npy"
        self.inputs_file_dst = "inputs.npy"

        self.outputs_file_src = "outputs.npy"
        self.outputs_file_dst = "outputs.npy"
        self.model_file_src = engine.file
        self.model_file_dst = "model.onnx"

        self.sock = InferenceEngineRemoteSocket()

    def serialize_inputs(self):
        data = np.stack(self.inputs)
        np.save(self.inputs_file_src, data)

    def deserialize_outputs(self, outputs_file):
        return np.load(outputs_file)

    def infer(self, x):
        self.inputs.append(x)

    def get_outputs(self):
        self.serialize_inputs()

        cmd = " && \\\n".join([
            f"kubectl cp {self.model_file_src} {self.pod}:{self.target_dir}/{self.model_file_dst}",
            f"kubectl cp {self.inputs_file_src} {self.pod}:{self.target_dir}/{self.inputs_file_dst}",
            f"kubectl exec {self.pod} -- {self.target_dir}/{self.server_script} --model {self.model_file_dst} --inputs {self.inputs_file_dst} --outputs {self.outputs_file_dst}",
            f"kubectl cp {self.pod}:{self.target_dir}/{self.outputs_file_dst} {self.outputs_file_src}",
        ])

        os.system(cmd)

        return self.deserialize_outputs(self.outputs_file)


class CalibrationDatasetBase:
    pass


class CalibrationDatasetImage(CalibrationDatasetBase):
    def __init__(self, img_path, needs_preproc, transform=None, limit=None) -> None:
        super().__init__()

        if img_path.endswith(".txt"):
            root_path = os.path.dirname(img_path)
            with open(img_path, "r") as f:
                img_files = [os.path.join(root_path, p.rstrip()) for p in f.readlines()]
        else:
            img_files = glob.glob(img_path)

        assert len(img_files) > 0
        assert os.path.exists(img_files[0])

        if limit is not None:
            random.Random(123).shuffle(img_files)
            img_files = img_files[:limit]

        self.img_files = img_files
        self.transform = transform
        self.needs_preproc = needs_preproc

    def __getitem__(self, i):
        x = cv2.cvtColor(cv2.imread(str(self.img_files[i])), cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x

    def __len__(self):
        return len(self.img_files)