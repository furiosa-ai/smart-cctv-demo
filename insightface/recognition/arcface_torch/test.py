import argparse
import os
from pathlib import Path
from typing import List, OrderedDict

import numpy as np
import torch

from backbones import get_model
from eval import verification
from bin_to_np import load_np
from ai_util.inference_framework import CalibrationDatasetImage, PredictorBase
from recognition.arcface_torch.utils.arcface_predictor import ArcFacePredictor
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."



class Validator(object):
    def __init__(self, val_targets, rec_prefix, image_size=(112, 112)):
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.batch_size = 1
        self.nfolds = 10
        self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module):
        global_step = 0
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, self.batch_size, self.nfolds)
            print('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            print(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = load_np(path, image_size)
                # data_set = [t.to("cuda") for t in data_set[0]], data_set[1]
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, backbone: torch.nn.Module):
        self.ver_test(backbone)


def main(args):
    # get config
    # torch.cuda.set_device(0)

    cfg = get_config(args.config, args)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    """
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size, act=cfg.act).eval()

    checkpoint = torch.load(args.weights, map_location="cpu")

    backbone.load_state_dict(checkpoint)
    print(f"Loaded checkpoint {args.weights}")
    """

    backbone = ArcFacePredictor(cfg=cfg, weights=args.weights, name=Path(args.config).stem)
    backbone.to("furiosa")

    validator = Validator(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec
    )

    validator(backbone)


if __name__ == "__main__":
    print("Start")
    # torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--weights", required=True)
    main(parser.parse_args())   