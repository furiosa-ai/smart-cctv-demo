import os

import numpy as np


class ImageBatchStream:
    def __init__(self, batch_size, calibration_files, input_shape, preprocessor=None, normalize_img=True):
        self.channels, self.height, self.width = input_shape

        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + \
                           (1 if (len(calibration_files) % batch_size) \
                            else 0)
        self.files = calibration_files
        self.calibration_data = np.zeros((batch_size, self.channels, self.height, self.width), \
                                         dtype=np.float32)
        self.batch = 0
        self.preprocessor = preprocessor
        self.normalize_img = normalize_img
        self.input_shape = input_shape

    def read_image_chw(self, path):
        from PIL import Image

        img = Image.open(path).resize((self.width, self.height), Image.NEAREST)

        im = np.array(img, dtype=np.float32, order='C')

        if len(im.shape) == 2:
            im = np.stack([im] * 3, axis=2)

        im = im[:, :, ::-1]
        im = im.transpose((2, 0, 1))

        if self.normalize_img:
            im /= 255.0

        assert len(im.shape) == 3

        return im

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch: \
                                         self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                # print("[ImageBatchStream] Processing ", f)
                img = self.read_image_chw(f)
                if self.preprocessor is not None:
                    img = self.preprocessor(img)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])
