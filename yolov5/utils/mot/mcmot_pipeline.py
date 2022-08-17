# from multiprocessing import Process, Lock, Queue
# from threading import Thread as Process, Lock; from queue import Queue

import numpy as np
from collections import namedtuple
import yaml
from calibration.util.camera_calib_node import CameraCalibrationNode
from utils.logging import log_func
from utils.mot.detector import Detector
from utils.mot.detector_model import DetectorModelPool
from utils.mot.mcmot import MCMOT
from utils.mot.mcmot_display import MCMOTDisplay
from utils.mot.mcmot_stub import StubDetector, StubDetectorModelPool, StubMCMOT
from utils.mot.mot_camera import MOTCamera

from utils.mot.mot_pipeline import MOTPipeline
from utils.mot.tracker import Tracker
from utils.mot.video_input import VideoInput
from utils.util import PerfMeasure, MPLib



# MOTTaskOutput = namedtuple("MOTTaskOutput", ("image", "mot_cam"))


class MCMOTPipeline:
    def __init__(self, cfg=None) -> None:
        if isinstance(cfg, str):
            with open(cfg, "r") as f:
                cfg = yaml.safe_load(f)

        self.cfg = cfg

        self.multiproc = self.cfg["system"]["multiproc"]
        self.frame_limit = self.cfg["system"].get("frame_limit", None)
        self.mot_count = len(self.cfg["mot"]["cameras"])

        self.is_running = True

        MPLib.set_strategy(self.multiproc)

        self.use_shm = self.cfg["system"].get("use_shm", False)

        self.deproj_freq = self.cfg.get("mcmot", {}).get("deproj_freq", 1)

    def _create_mcmot(self):
        if self.cfg.get("mcmot", {}).get("stub", False):
            return StubMCMOT()
        else:
            return MCMOT(self.mot_count, **self.cfg.get("mcmot", {}))

    def _create_display(self):
        return MCMOTDisplay(mot_count=self.mot_count, **self.cfg["display"])

    def _create_mot_task(self, idx, det_model, mot_input_queue, mot_output_queue):
        proc = MPLib.Process(target=self._mot_task, args=(idx, det_model, mot_input_queue, mot_output_queue), name=f"MOT{idx}")
        return proc

    def _create_mot_pipeline(self, idx, det_model):
        # cfg = self.cfg["mot"][idx]
        cfg = self.cfg["mot"]
        cam_cfg = cfg["cameras"][idx]

        # det_model = DetectorModelPool(**cfg["detector"]["model"])

        mot_pipeline = MOTPipeline(
            video_input=VideoInput(**cam_cfg["input"], frame_limit=self.frame_limit),
            detector=self._create_detector(det_model),
            tracker=Tracker(**cfg["tracker"]),
            calib=CameraCalibrationNode.from_file(cam_cfg["calib"], undist=cam_cfg.get("undistort", False)),
            proj3d=cfg.get("proj3d", False),
            box_smoothing=cfg.get("box_smoothing", "default"),
            traj_smoothing=cfg.get("traj_smoothing", "default"),
        )

        return mot_pipeline

    def _create_detector(self, det_model):
        if not self.cfg["mot"]["detector"].get("stub", False):
            return Detector(det_model=det_model, **self.cfg["mot"]["detector"]["postproc"])
        else:
            return StubDetector()

    def _create_detector_model_pool(self):
        if not self.cfg["mot"]["detector"].get("stub", False):
            return DetectorModelPool(inference_count=self.mot_count, use_shm=self.use_shm, **self.cfg["mot"]["detector"]["model"])
        else:
            return StubDetectorModelPool()

    def _mot_task(self, idx, det_model, input_queue, output_queue):
        #init

        #while
        #acq mot lock

        mot_pipeline = self._create_mot_pipeline(idx, det_model)

        # if True is in the queue run otherwise quit
        while input_queue.get(block=True):
            # self.mot_process_lock[idx].acquire(True)
            # lk.acquire(True)
            print(f"MOT {idx}: Running MOT")

            mot_res = mot_pipeline()

            # self.mot_results[idx] = mot_pipeline()

            # print(f"MOT {idx}: Putting output in queue...")
            output_queue.put(mot_res, block=False)  # no need to block, already handled by lock

            # with self.mots_finished_cv:
            #     self.mots_finished_cv.notify()

            # print(f"MOT {idx}: Waiting for next signal")

        print(f"MOT {idx} exited")

    def calibrate(self):
        pass

    @log_func
    def _add_det(self, mot_cams, mot_results, frame_idx):
        for mot_cam, mot_res in zip(mot_cams, mot_results):
            mot_cam.add_det(mot_res.boxes_track, mot_res.deproj_points if frame_idx % self.deproj_freq == 0 else None)

    @log_func
    def _mcmot_step(self, mot_input_queues, mot_output_queues, mot_cams, mcmot, display, frame_idx):
        is_running = self.is_running

        print(f"Waiting for MOTs")

        """
        with self.mots_finished_cv:
            # unblock local mots
            for i in range(self.mot_count):
                self.mot_process_lock[i].release()
            self.mots_finished_cv.wait_for(lambda: all(res != None for res in self.mot_results))
        """

        # wait for all mots to produce a result
        mot_results = [qu.get(block=True) for qu in mot_output_queues]

        if all(res.image is None for res in mot_results):
            is_running = False

        if self.frame_limit is not None and frame_idx >= self.frame_limit - 1:
            is_running = False

        # all results acquired -> mots can run again
        for qu in mot_input_queues:
            qu.put(is_running)

        print(f"MOT finished. Running MCMOT")
        self._add_det(mot_cams, mot_results, frame_idx)
        mcmot(mot_cams)

        with PerfMeasure("display"):
            display(mcmot, mot_cams, mot_results)

        return is_running

    def run(self):
        # for i in range(self.mot_count):
        #     self.mot_process_lock[i].acquire()

        # for lk in self.mot_process_lock:
        #         lk.acquire()

        if self.use_shm:
            MOTOutputQueues = []

            for cam_cfg in self.cfg["mot"]["cameras"]:
                mot_out_fmt, mot_out_shm_supported = MOTPipeline.get_output_format(cam_cfg)
                MOTOutputQueues.append(MPLib.create_dyn_queue(mot_out_fmt, mot_out_shm_supported))
        else:
            MOTOutputQueues = [MPLib.Queue] * self.mot_count

        mot_input_queues = [MPLib.Queue(1) for _ in range(self.mot_count)]
        mot_output_queues = [Qu(1) for Qu in MOTOutputQueues]

        det_model_pool = self._create_detector_model_pool()

        mot_cams = [MOTCamera(
            traj_timeout=self.cfg["mot"].get("local_traj_timeout", np.inf),
            max_traj_length=self.cfg["mot"].get("max_traj_length", None),
            comp_trajs=self.cfg["mot"]["cameras"][i].get("comp_trajs", True),
            smooth_timeout=self.cfg["mot"].get("smooth_timeout", False),
            max_traj_count=self.cfg["mot"].get("max_traj_count", None),
        ) for i in range(self.mot_count)]

        mcmot = self._create_mcmot()
        mot_procs = [
            self._create_mot_task(idx, det_model_pool.get_inference(idx), mot_input_queues[idx], mot_output_queues[idx]) 
            for idx in range(self.mot_count)
        ]

        for mot_proc in mot_procs:
            mot_proc.start()

        det_model_pool.start()

        # Need to somehow create matplotlib plots after all processes are started
        display = self._create_display()

        self.is_running = True
        for qu in mot_input_queues:
            qu.put(self.is_running)

        frame_idx = 0
        while self._mcmot_step(mot_input_queues, mot_output_queues, mot_cams, mcmot, display, frame_idx):
            frame_idx += 1

        # exit

        print("MCMOT finished. Exiting...")

        display.exit()
        print("MCMOT display exited")

        det_model_pool.exit()
        print("Model pool exited")

        for proc in mot_procs:
            proc.join()
        print("MCMOT exited")

    def exit(self):
        self.is_running = False

def _test():
    mcmot_pipeline = MCMOTPipeline(cfg="cfg/mot_test_pipeline.yaml")
    mcmot_pipeline.run()


if __name__ == "__main__":
    _test()
