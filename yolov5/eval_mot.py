from turtle import forward
import torch
from torch import nn
import sys
import os
import utils
import utils.load_images

from utils.model_conversion import CalibrationDataPipeline, model_to_device

sys.path.append("bytetrack")

from models.common import DetectMultiBackend
from utils.general import non_max_suppression


from bytetrack.yolox.data import MOTDataset, ValTransform
from bytetrack.yolox.evaluators import COCOEvaluator
from models.yolo import Model

from bytetrack.tools.track import make_parser, get_exp, launch, main


class Yolov5Predictor(nn.Module):
    def __init__(self, model_cfg, weights, device) -> None:
        super().__init__()

        Model(model_cfg) # load global config
        self.model = DetectMultiBackend(weights, device=device, dnn=False)
        self.device = device

    def forward(self, x):
        pred = self.model(x)
        return pred


class Yolov5FuriosaPredictor(nn.Module):
    def __init__(self, model_cfg, weights, imgsz) -> None:
        super().__init__()

        convert_params = {
            "input_shape": (3, *imgsz),
            "output_names": ("feat1", "feat2", "feat3"),
            "calib_data": CalibrationDataPipeline([
                utils.load_images.LoadImages("../data/coco/images/val2017", output_dict=False, resize=(imgsz[1], imgsz[0]), as_tensor=False, img_count=100), 
            ], model_in_name="input"), 
            "model_path": "out/",
            "model_name": "yolov5",    
            "use_cache": True,
            # "opset": 12
        }

        Model(model_cfg) # load global config
        detect = DetectMultiBackend(weights, device="cpu", dnn=False)

        model = detect.model
        det_layer = model.model[-1]
        det_layer.set_mode("export")
        model = model_to_device(model, "furiosa", **convert_params)

        det_layer.set_mode("decode")
        model = torch.nn.Sequential(model, det_layer)

        self.detect = model
        self.device = "furiosa"

    def forward(self, x):
        pred = self.detect(x)
        return pred


class Yolov5Exp:
    def __init__(self, model_cfg, weights, dataset_name, device) -> None:

        self.output_dir = "out/"
        self.experiment_name = "exp"
        self.test_size = (896, 1600)
        # self.test_size = (128, 128)
        # self.test_size = (32, 32)
        # self.test_size = (672, 1216)
        self.val_ann = "val_half.json" 
        self.data_num_workers = 4
        self.num_classes = 1
        self.dataset_name = dataset_name

        if device == "furiosa":
            model = Yolov5FuriosaPredictor(model_cfg, weights, imgsz=self.test_size)
        else:
            model = Yolov5Predictor(model_cfg, weights, device=device)

        self.model = model

    def get_model(self):
        return self.model

    """
    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
    """

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        valdataset = MOTDataset(
            data_dir=f"../data/{self.dataset_name}",
            json_file=self.val_ann,
            img_size=self.test_size,
            name='train', # change to train when running on training set
            preproc=ValTransform(
                rgb_means=(0, 0, 0),
                std=(1, 1, 1),
                # rgb_means=(0.485, 0.456, 0.406),
                # std=(0.229, 0.224, 0.225),
            ),
        )

        """
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
        """
        sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader


def _main():
    args = make_parser().parse_args()
    exp = Yolov5Exp(args.model_cfg, args.ckpt, args.data, args.device)
    # exp = None

    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    # num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    # assert num_gpu <= torch.cuda.device_count()
    num_gpu = 1

    main(exp, args=args, num_gpu=num_gpu, skip_ckpt_load=True)

    """
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
    """


if __name__ == "__main__":
    _main()
