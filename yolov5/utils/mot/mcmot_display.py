from math import ceil
import cv2
import numpy as np
from calibration.util.axes_plot_3d import AxesPlot3d
from utils.colors import colors as _colors
from utils.fps_counter import FpsCounter
from utils.logging import log_func
from utils.mot.video_input import OpenCVVideoRecord
from utils.util import PerfMeasure

class MCMOTDisplay:
    def __init__(self, mot_count, plot_cols, size_per_mot, show_vid=True, record_vid=False, record_vid_raw=False, show_plot=True, *args, **kwargs) -> None:
        plot_count = mot_count + 2
        num_col = 4
        num_row = ceil(plot_count / num_col)
        # *ax_local, ax_local_all, ax_global = [AxesPlot3d(num_col=4, num_row=num_row) for i in range(plot_count)] 

        self.mot_count = mot_count

        self.plot_count = plot_count
        self.cols = plot_cols
        self.rows = ceil(self.mot_count / self.cols)
        self.colors = _colors
        self.size_per_mot = tuple(size_per_mot)
        self.imgs = None
        self.img_grid = self._create_grid(size_per_mot)
        # self.ax_local = ax_local
        # self.ax_local_all = ax_local_all
        # self.ax_global = ax_global

        self.empty_img = np.zeros((self.size_per_mot[1], self.size_per_mot[0], 3), dtype=np.uint8)

        self.show_vid = show_vid
        self.record_vid = record_vid
        self.record_vid_raw = record_vid_raw
        self.show_plot = show_plot

        self.video_writer = OpenCVVideoRecord("out/out.mp4", fps=10, start_rec=True)
        self.video_writer_raw = [OpenCVVideoRecord(f"out/cam{i}.mp4", fps=10, start_rec=True) for i in range(mot_count)]

        self.fps_counter = FpsCounter()
        self.fps = 0

    def _create_grid(self, size_per_mot):
        w, h = size_per_mot
        w *= self.cols
        h *= self.rows

        return np.zeros((h, w, 3), dtype=np.uint8)

    @log_func
    def _draw_boxes(self, img, boxes):
        ih, iw = img.shape[:2]
        for *box, track_id in boxes:
            # if box is not None:
            box = np.array(box)

            box[0::2] *= iw
            box[1::2] *= ih

            x1, y1, x2, y2 = box.astype(int)
            r, g, b = self.colors(track_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), 3)
            cv2.putText(img, "{}".format(int(track_id)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (b, g, r), 2)

        return img

    def _resize_img(self, img, size):
        img_res = cv2.resize(img, size)
        scale = img_res.shape[1] / img.shape[1]
        return img_res, scale

    def _make_grid(self, imgs):
        w, h = self.size_per_mot

        for i, img in enumerate(imgs):
            x = i % self.cols
            y = i // self.cols
            
            x1 = w * x
            y1 = h * y
            x2 = w * (x + 1)
            y2 = h * (y + 1)

            self.img_grid[y1:y2, x1:x2] = img

        cv2.putText(self.img_grid, f"{self.fps:.1f}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def _get_pylplot_colors(self, color_ids):
        return [np.array(self.colors(i))[::-1] / 255 for i in color_ids]

    @log_func
    def draw_plot(self, mcmot, mot_cams):
        """
        for ax_local in self.ax_local:
            ax_local.clear()
        self.ax_local_all.clear()
        self.ax_global.clear()


        for i in range(len(mot_cams)):
            trajs_loc, color_ids_loc = mcmot.query_trajs(mot_cams, track_type="local", color_by="global_id", cam_indices=[i])
            self.ax_local[i].plot_all(trajs_loc, colors=self._get_pylplot_colors(color_ids_loc))
        trajs_loc_all, color_ids_loc_all = mcmot.query_trajs(mot_cams, track_type="local", color_by="global_id")
        trajs_glob, color_ids_glob = mcmot.query_trajs(mot_cams, track_type="global", color_by="global_id")

        self.ax_local_all.plot_all(trajs_loc_all, colors=self._get_pylplot_colors(color_ids_loc_all))
        self.ax_global.plot_all(trajs_glob, colors=self._get_pylplot_colors(color_ids_glob))

        # trajs = trajs_loc + trajs_loc_all + trajs_glob
        # color_ids = color_ids_loc + color_ids_loc_all + color_ids_glob
        # color_loc, color_loc_all, color_glob = [[np.array(self.colors(i))[::-1] / 255 for i in color_ids] 
        #     for color_ids in [color_ids_loc, color_ids_loc_all, color_ids_glob]]

        # for ax_local in self.ax_local:
        #     ax_local.plot_all(trajs_loc, colors=color_loc)
        # self.ax_local_all.plot_all(trajs_loc_all, colors=color_loc_all)
        # self.ax_global.plot_all(trajs_glob, colors=color_glob)

        # only need to call once
        self.ax_global.draw()
        self.ax_global.show(pause=0.001)
        """
        pass

    @log_func
    def _record_frame_raw(self, cam_idx, frame):
        if frame is None:
            frame = np.zeros((3, self.video_writer_raw[cam_idx].size[1], self.video_writer_raw[cam_idx].size[0]), dtype=np.uint8)

        self.video_writer_raw[cam_idx].update(frame)

    @log_func
    def _record_frame(self, frame):
        self.video_writer.update(frame)

    @log_func
    def draw_boxes(self, mcmot, mot_cams, mot_results):
        cam_imgs = []
        for cam_idx, mot_cam in enumerate(mot_cams):
            img = mot_results[cam_idx].image

            if img is not None:
                boxes = mcmot.get_boxes(cam_idx, mot_cam)  # with global id, rel coords

                img, scale = self._resize_img(img, self.size_per_mot)

                self._draw_boxes(img, boxes)

                # draw fps
                # if cam_idx == 0:
                    
            else:
                img = self.empty_img

            cam_imgs.append(img)

        self.imgs = cam_imgs

    @log_func
    def display_cameras(self):
        cv2.imshow("out", self.img_grid)
        cv2.waitKey(1)

    @log_func
    def __call__(self, mcmot, mot_cams, mot_results):
        self.fps = self.fps_counter.step()
        print(f"FPS: {self.fps:.1f}")

        if self.show_vid and self.show_plot:
            with PerfMeasure("Plotting"):
                self.draw_plot(mcmot, mot_cams)

        if self.record_vid_raw:
            for cam_idx, mot_cam in enumerate(mot_cams):
                img = mot_results[cam_idx].image

                if self.record_vid_raw:
                    self._record_frame_raw(cam_idx, img)

        if self.show_vid or self.record_vid:
            self.draw_boxes(mcmot, mot_cams, mot_results)
            self._make_grid(self.imgs)

        if self.show_vid:
            self.display_cameras()

        if self.record_vid:
            self._record_frame(self.img_grid)

    def exit(self):
        self.video_writer.close()

        for vw in self.video_writer_raw:
            vw.close()
