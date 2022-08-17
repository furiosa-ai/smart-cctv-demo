
import cv2
import numpy as np


class ImageGalleryReader:
    def __init__(self, gallery, img_size=None, vis_best_only=False) -> None:
        self.gallery = gallery
        self.data = gallery.data
        self.key_idx_map = gallery.key_idx_map
        self.keys = list(self.key_idx_map.keys())
        self.distmat = None
        self.distmat_idx = 0
        self.img_size = img_size
        self.vis_best_only = vis_best_only

    def update_settings(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    def draw(self, img, entries, dists, scale):
        return img

    def set_distmat(self, distmat):
        self.distmat = distmat

    def get_fps(self):
        return self.data.get_fps()

    def __len__(self):
        return len(self.data)

    def open(self):
        self.data.open()

    def close(self):
        self.data.close()

    # def set_frame_key(self, key):
    #     idx = self.keys.index(key)
    #     self.next_idx = idx

    def set_frame_idx(self, idx):
        self.data.set_frame_idx(idx)

    def get_frame_idx(self):
        return self.data.get_frame_idx()

    def get_frame_count(self):
        return len(self)

    def _find_closest_key(self, idx, max_offset=2):
        for offset in range(max_offset+1):
            if (idx + offset) in self.key_idx_map:
                return idx + offset
            elif (idx - offset) in self.key_idx_map:
                return idx - offset

        # raise Exception("Cant find closest key")
        return None

    def read(self):
        # order important
        idx = self.get_frame_idx()
        img = self.data.read()

        # if idx not in self.key_idx_map:
        #     return img

        idx = self._find_closest_key(idx, max_offset=self.gallery.key_freq // 2)

        if self.img_size is not None:
            scale = self.img_size / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            scale = 1

        if idx is not None:
            ind = self.key_idx_map[idx]

            entries = [self.gallery[i] for i in ind]

            if self.distmat is not None:
                dists = self.distmat[self.distmat_idx, ind]

                if self.vis_best_only and len(entries) > 0:
                    best_idx = np.argmin(dists)

                    entries = [entries[best_idx]]
                    dists = [dists[best_idx]]
            else:
                dists = None

            img = self.draw(img, entries, dists, scale)

        return img