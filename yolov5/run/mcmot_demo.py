import argparse
from utils.logging import logger
import multiprocessing
import platform
import cv2
from types import SimpleNamespace
import yaml


def dict_to_ns(d):
    x = SimpleNamespace()
    _ = [setattr(x, k, dict_to_ns(v)) if isinstance(v, dict) else setattr(x, k, v) for k, v in d.items() ]    
    return x


def ns_to_dict(ns):
    d = {}

    for k, v in ns.__dict__.items():
        if isinstance(v, SimpleNamespace):
            d[k] = ns_to_dict(v)
        else:
            d[k] = v

    return d


def _dict_override(dic, k, v):
    k = k.split(".", 1)
    if len(k) == 1:
        if v is not None or k[0] not in dic:
            dic[k[0]] = v
    else:
        if k[0] not in dic:
            dic[k[0]] = {}
        _dict_override(dic[k[0]], k[1], v)


def update_with_args(dic, args):
    '''
    Pass args from ArgParse to override config values
    :param args: Args from ArgParse
    '''
    # new_vals = {k: v for k, v in vars(args).items() if v is not None}
    # self.vals.update(new_vals)

    for k, v in vars(args).items():
        _dict_override(dic, k, v)


def load_cfg(cfg_file, args):
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    update_with_args(cfg, args)

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--gui", default="qt", choices=("cv", "qt"))
    # parser.add_argument("--record", dest="display.record_vid", action="store_true")
    # parser.add_argument("--record_raw", dest="display.record_vid_raw", action="store_true")
    # parser.add_argument("--hide_vid", dest="display.show_vid", action="store_false")
    # parser.add_argument("--hide_plot", dest="display.show_plot", action="store_false")
    parser.add_argument("--device", dest="mot.detector.model.device")
    parser.add_argument("--max_traj_count", dest="mot.max_traj_count", type=int)
    parser.add_argument("--frame_limit", dest="system.frame_limit", type=int)
    parser.add_argument("--det_stub", dest="mot.detector.stub", action="store_true")
    # parser.add_argument("--mcmot_stub", dest="mcmot.stub", action="store_true")
    parser.add_argument("--mcmot_matching_thresh", dest="mcmot.matching_thresh", type=float)
    # parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    print(f"Uses optimized OpenCV: {cv2.useOptimized()}")

    cfg = load_cfg(args.cfg, args)

    if args.gui == "cv":
        from utils.mot.mcmot_pipeline import MCMOTPipeline
        mcmot_pipeline = MCMOTPipeline(cfg=cfg)
        mcmot_pipeline.run()
    elif args.gui == "qt":
        logger.start()

        from utils.mot.mcmot_app_qt import run_mcmot_qt
        run_mcmot_qt(cfg=cfg)
    else:
        raise Exception(args.gui)

    logger.exit()


if __name__ == "__main__":
    main()
