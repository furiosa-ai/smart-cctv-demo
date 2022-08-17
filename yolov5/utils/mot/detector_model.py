# from multiprocessing import Process, Lock, Queue, Condition
from collections import namedtuple
from contextlib import ExitStack
import copy
import os
import numpy as np
import torch
from models.common import DetectMultiBackend
from models.yolo import Model
from utils.logging import log_func
from utils.model_conversion import CalibrationDataPipeline, model_to_device

from utils.torch_utils import select_device
# from threading import Thread as Process, Lock; from queue import Queue

import utils.load_images
from utils.util import MPLib, PerfMeasure


ModelInput = namedtuple("ModelInput", ("inf_idx", "inputs"))
ModelOutput = namedtuple("ModelOutput", ("outputs"))


class DetectorModel:
    def __init__(self, device_idx, config, weights, device, input_size, batch_size, input_type="f32", input_format="chw") -> None:
        if device is not None and (device.startswith("onnx") or device == "furiosa"):
            data_device = "cpu"
        elif device == "cuda":
            data_device = f"cuda:{device_idx}"
        else:
            data_device = select_device(device)

        Model(config)  # to load global config
        model = DetectMultiBackend(weights, device=data_device, dnn=False, data=None)
        model.eval()

        model_name = os.path.splitext(weights)[0].replace("/", "_")

        if device != "furiosa":
            input_type = "f32"
            input_format = "chw"

        # TODO: resize will stretch image during calibration -> use crop
        convert_params = {
            "input_shape": (3, input_size[1], input_size[0]),
            "output_names": ("feat1", "feat2", "feat3"),
            "calib_data": CalibrationDataPipeline([
                utils.load_images.LoadImages("datasets/calib/coco", output_dict=False, resize=(input_size[0], input_size[1]), as_tensor=False, img_count=100), 
            ], model_in_name="input", batch_size=batch_size), 
            "model_path": "out/",
            "model_name": model_name,    
            "use_cache": True,
            "device_name": device_idx,
            "batch_size": batch_size,
            "input_type": input_type,
            "input_format": input_format
            # "opset": 12
        }

        # infer_dev = "onnx_i8"
        # infer_dev = "furiosa"
        # infer_dev = None
        # infer_dev = None

        model = model.model
        det_layer = model.model[-1]

        det_layer.set_mode("export")

        # split yolo detect layer in two parts (conv+sigmoid "export" and box decode)
        if device is not None and (device.startswith("onnx") or device == "furiosa"):
            model = model_to_device(model, device, **convert_params)

        self.model = model
        self.data_device = data_device
        self.device_idx = device_idx
        self.input_type = input_type
        self.input_format = input_format

    @log_func
    def _infer(self, img_tensor):
        return self.model(img_tensor)

    def __call__(self, img_tensor):
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor)

        img_tensor = img_tensor.to(self.data_device)
        # with PerfMeasure(f"Model inference {self.device_idx}"):
        outputs = self._infer(img_tensor)
        outputs = [o.cpu().numpy() for o in outputs]

        return outputs


class BatchedQueue:
    def __init__(self, batch_size, max_qu_size, use_shm=False, input_fmt=None, input_shm_supported=None) -> None:
        # dont use lock if batch size is 1

        if use_shm:
            Qu = MPLib.create_dyn_queue(input_fmt, input_shm_supported, safe_data_access=False)
        else:
            Qu = MPLib.Queue

        self.queue = Qu(max_qu_size)
        self.lk = MPLib.Lock() if batch_size > 1 else None
        self.batch_size = batch_size

    def get(self, *args, **kwargs):
        # prevent stealing missing elements and give all elements to currently requesting thread
        with ExitStack() as stack:
            if self.lk is not None:
                stack.enter_context(self.lk)
            
            return [self.queue.get(*args, **kwargs) for _ in range(self.batch_size)]

    def put(self, item, *args, **kwargs):
        self.queue.put(item, *args, **kwargs)


# manages same model over multiple inference devices
class ModelPool(object):
    def __init__(self, inference_count, instance_count, batch_size=1, use_shm=False, *args, **kwargs) -> None:
        self.batch_size = batch_size
        self.inference_count = inference_count
        self.instance_count = instance_count
        self.model_args = args
        self.model_kwargs = kwargs

        model_input_fmt, model_input_shm_supported = self.get_input_format()

        self.input_queue = BatchedQueue(batch_size=self.batch_size, max_qu_size=inference_count, 
            use_shm=use_shm, input_fmt=model_input_fmt, input_shm_supported=model_input_shm_supported)
        self.output_queues = [MPLib.Queue(1) for _ in range(self.inference_count)]  # TODO: try shm here too
        self.model_create_lk = MPLib.Lock()
        self.procs = None

    def get_input_format(self):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def _create_model(self, idx, *args):
        raise NotImplementedError()

    def _create_model_task(self, idx, *args, **kwargs):
        proc = MPLib.Process(target=self._model_task, args=(idx, self.model_create_lk, self.input_queue, self.output_queues, *args), 
            kwargs=kwargs, name=f"{self.get_name()}{idx}")
        proc.start()
        return proc

    def _model_task(self, idx, model_create_lk, input_queue, output_queues, *args, **kwargs):
        # only create one model at a time -> create in first thread, load cached file in other threads
        with model_create_lk:
            model = self._create_model(idx, *args, **kwargs)
        output_queues[0].put(True)  # notify that model creation has been completed

        while True: # self.is_running:
            inputs = self._recv_model_input(input_queue)
            if inputs is not None and inputs[0] is not None:
                inf_ind, model_inputs_list = zip(*inputs)
                input_batch = np.concatenate(model_inputs_list)
                outputs_batch = model(input_batch)

                # dissolve batch
                for i, inf_idx in enumerate(inf_ind):
                    if isinstance(outputs_batch, (tuple, list)):
                        out = [o[i:i+1] for o in outputs_batch]
                    else:
                        out = outputs_batch[i:i+1]
                    self._send_model_output(output_queues, inf_idx, out)
            else:
                break

    """
    @staticmethod
    def create(self, count, *args, **kwargs):
        return [DetectorModelPool(*args, **kwargs) for _ in count]
    """

    # @log_func
    def _recv_model_input(self, input_queue):
        return input_queue.get(block=True)

    # @log_func
    def _send_model_output(self, output_queues, inf_idx, out):
        output_queues[inf_idx].put(ModelOutput(out))

    # @log_func
    def _send_model_input(self, inference_idx, model_inputs):
        self.input_queue.put(ModelInput(inference_idx, model_inputs))

    # @log_func
    def _recv_model_output(self, inference_idx):
        return self.output_queues[inference_idx].get(block=True)

    @log_func
    def __call__(self, inference_idx, model_inputs):
        self._send_model_input(inference_idx, model_inputs)
        output, = self._recv_model_output(inference_idx)

        return output

    def start(self):
        self.procs = [self._create_model_task(idx, *self.model_args, **self.model_kwargs) for idx in range(self.instance_count)]

        # wait until all models are built
        for _ in range(self.instance_count):
            assert self.output_queues[0].get(block=True)

    def exit(self):
        # self.is_running = False
        # need to put something in input queue so _model_task can finish

        for _ in range(len(self.procs)):
            self.input_queue.put(None)

        for proc in self.procs:
            proc.join()

    def get_inference(self, idx):
        assert idx < self.inference_count
        return ModelPoolInference(self, idx)


class ModelPoolInference(object):
    def __init__(self, pool, idx) -> None:
        self.pool = pool
        self.idx = idx

    """
    def __getattr__(self, __name: str):
        return getattr(self.pool, __name)
    """

    def __call__(self, model_inputs):
        return self.pool(self.idx, model_inputs)


class DetectorModelPool(ModelPool):
    def __init__(self, weights=None, device=None, input_size=None, input_type="f32", input_format="chw", *args, **kwargs) -> None:

        if device != "furiosa":
            input_type = "f32"
            input_format = "chw"

        super().__init__(*args, **kwargs, weights=weights, input_size=input_size, input_type=input_type, input_format=input_format, device=device)

    def get_input_format(self):
        w, h = self.model_kwargs["input_size"]
        input_shape = (self.batch_size, 3, h, w) if self.model_kwargs["input_format"] == "chw" else (self.batch_size, h, w, 3)

        inf_idx = None
        inputs = np.zeros(input_shape, dtype=np.uint8 if not self.model_kwargs["input_type"] == "f32" else np.float32)

        return ModelInput(
            inf_idx=inf_idx,
            inputs=inputs
        ), [False, True]

    def get_name(self):
        return "DET"

    def _create_model(self, idx, *args, **kwargs):
        return DetectorModel(device_idx=idx, batch_size=self.batch_size, *args, **kwargs)

    def get_inference(self, idx):
        infer = super().get_inference(idx)

        for k, v in self.model_kwargs.items():
            setattr(infer, k, v)

        return infer
