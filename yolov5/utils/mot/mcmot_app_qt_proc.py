import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QObject, QEvent
from collections import namedtuple
from utils.logging import log_func
from utils.mot.mcmot_app_qt_utils import MCMOTQTSignal
from utils.mot.mcmot_display import MCMOTDisplay
from utils.mot.mcmot_pipeline import MCMOTPipeline
from utils.util import MPLib, PerfMeasure


MCMOTQTDisplayOut = namedtuple("MCMOTQTDisplayOut", ("frames", "boxes", "plot_names", "plot_trajs", "plot_traj_ids", "fps"))

class MCMOTQTDisplayChildProcess(MCMOTDisplay):
    def __init__(self, out_qu, plot_names, cam_res, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_qu = out_qu
        self.plot_names = plot_names
        self.frame_idx = 0


        self.cur_data = None
        self.cam_res = cam_res

        # assert self.record_vid is False, "Not working anymore, boxes are drawn in gl"

    @staticmethod
    def _create_empty_out_data(mot_count, cam_res, fps=None):
        return MCMOTQTDisplayOut(
            frames=np.zeros((mot_count, cam_res[1], cam_res[0], 3), dtype=np.uint8),
            boxes=[],
            plot_names=[],
            plot_trajs=[],
            plot_traj_ids=[],
            fps=fps
        )

    def _create_out_data(self, fps=None):
        self.cur_data = self._create_empty_out_data(self.mot_count, self.cam_res, fps)

    @staticmethod
    def get_output_format(mot_count, cam_res):
        return MCMOTQTDisplayChildProcess._create_empty_out_data(mot_count, cam_res, None), [True, False, False, False, False, False]

    def _send_data(self):
        self.out_qu.put(self.cur_data)

    def display_cameras(self, imgs, boxes):
        for i, (img, box) in enumerate(zip(imgs, boxes)):
            self.cur_data.frames[i] = img
            self.cur_data.boxes.append(box)

    def _plot(self, plot_name, trajs, traj_ids):
        self.cur_data.plot_names.append(plot_name)
        self.cur_data.plot_trajs.append(trajs)
        self.cur_data.plot_traj_ids.append(traj_ids)
        # plot_data = self.qt_signal.parent.create_plot_data(plot_name, trajs, traj_ids)
        # self.qt_signal.plot.emit(plot_name, plot_data)

    @log_func
    def draw_plot(self, mcmot, mot_cams):
        # if self.frame_idx % 2 != 0:
        #     return

        with PerfMeasure("Querying plot data"):
            for i in range(len(mot_cams)):
                plot_name = f"local_{i}"

                if plot_name in self.plot_names:
                    trajs_loc, traj_ids_loc = mcmot.query_trajs(mot_cams, track_type="local", color_by="global_id", cam_indices=[i])
                    self._plot(plot_name, trajs_loc, traj_ids_loc)

            if "local_all" in self.plot_names:
                trajs_loc_all, traj_ids_loc_all = mcmot.query_trajs(mot_cams, track_type="local", color_by="global_id")
                self._plot("local_all", trajs_loc_all, traj_ids_loc_all)

            if "global" in self.plot_names:
                trajs_glob, traj_ids_glob = mcmot.query_trajs(mot_cams, track_type="global", color_by="global_id")
                self._plot("global", trajs_glob, traj_ids_glob)

    def __call__(self, mcmot, mot_cams, mot_results):
        self.fps = self.fps_counter.step()
        print(f"FPS: {self.fps:.1f}")

        self._create_out_data(self.fps)

        if self.show_vid and self.show_plot:
            self.draw_plot(mcmot, mot_cams)

        if self.record_vid_raw:
            for cam_idx, mot_cam in enumerate(mot_cams):
                img = mot_results[cam_idx].image

                if self.record_vid_raw:
                    self._record_frame_raw(cam_idx, img)

        # will be drawn in gl now
        # if self.show_vid or self.record_vid:
        #     self.draw_boxes(mcmot, mot_cams, mot_results)

        if self.show_vid:
            imgs, boxes = zip(*[(res.image, mcmot.get_boxes(cam_idx, mot_cam)) for cam_idx, (res, mot_cam) in enumerate(zip(mot_results, mot_cams))])

            # self.draw_boxes(mcmot, mot_cams, mot_results)
            # imgs = self.imgs

            self.display_cameras(imgs, boxes)

        self._send_data()

        if self.record_vid:
            self.draw_boxes(mcmot, mot_cams, mot_results)
            self._make_grid(self.imgs)
            self._record_frame(self.img_grid)

        self.frame_idx += 1


class MCMOTQTDisplayParentProcess:
    def __init__(self, qt_signal, in_qu) -> None:
        self.qt_signal = qt_signal
        self.in_qu = in_qu
        self.is_running = True

    def run(self):
        while self.is_running:
            d = self.in_qu.get(block=True)

            if d is None:
                break

            self.qt_signal.fps.emit(d.fps)

            for i, (img, box) in enumerate(zip(d.frames, d.boxes)):
                self.qt_signal.mot.emit(i, img, box)

            for plot_name, trajs, traj_ids in zip(d.plot_names, d.plot_trajs, d.plot_traj_ids):
                plot_data = self.qt_signal.parent.create_plot_data(plot_name, trajs, traj_ids)
                self.qt_signal.plot.emit(plot_name, plot_data)

    def exit(self):
        self.in_qu.put(None)


class MCMOTPipelineQTProcess(MCMOTPipeline):
    def __init__(self, cfg, out_qu, plot_names) -> None:
        super().__init__(cfg)
        self.out_qu = out_qu
        self.plot_names = plot_names

        cam_res = [cam["input"]["size"] for cam in cfg["mot"]["cameras"]]
        assert all((cam_res[0] == r for r in cam_res))
        self.cam_res = cam_res[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle qt_signal
        # del state["qt_signal"]
        return state

    def _create_display(self):
        return MCMOTQTDisplayChildProcess(out_qu=self.out_qu, plot_names=self.plot_names,
            mot_count=self.mot_count, cam_res=self.cam_res, **self.cfg["display"])

    @staticmethod
    def create_run(*args, **kwargs):
        pipeline = MCMOTPipelineQTProcess(*args, **kwargs)
        pipeline.run()


class MCMOTPipelineProcessQThread(QThread):
    qt_signal = MCMOTQTSignal()

    def __init__(self, pipeline_cls=None, cfg=None, parent=None) -> None:
        super().__init__(parent)

        if pipeline_cls is None:
            pipeline_cls = MCMOTPipelineQTProcess

        self.cfg = cfg
        self.pipeline_cls = pipeline_cls
        self.use_shm = cfg["system"].get("use_shm", False)

    def run(self):
        if self.use_shm:
            mot_count = len(self.cfg["mot"]["cameras"])
            cam_res = self.cfg["mot"]["cameras"][0]["input"]["size"]
            fmt, shm = MCMOTQTDisplayChildProcess.get_output_format(mot_count, cam_res)
            Qu = MPLib.create_dyn_queue(fmt, shm)
        else:
            Qu = MPLib.Queue

        qu = Qu(1)
        plot_names = self.qt_signal.plot_names
        proc = MPLib.Process(target=MCMOTPipelineQTProcess.create_run, kwargs=dict(cfg=self.cfg, out_qu=qu, plot_names=plot_names))
        self.display_main_proc = MCMOTQTDisplayParentProcess(self.qt_signal, qu)
        proc.start()
        self.display_main_proc.run()

        # self.pipeline.exit()
        proc.join()
        self.qt_signal.finished.emit()
    
    def exit_mcmot(self):
        self.display_main_proc.exit()