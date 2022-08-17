

def crop_center(img, crop_size):
    cropx, cropy = crop_size
    y, x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def tflite_image_representative_dataset(input_size, image_files):
    import tensorflow as tf
    import cv2
    import numpy as np
    from tqdm import tqdm

    # image_files = glob.glob("/home/kevin/Documents/projects/data/FDDB/calib/*")

    my_ds = tf.data.Dataset.range(len(image_files))

    # POST TRAINING QUANTIZATION
    def data_gen():
        for input_value in tqdm(my_ds.take(len(image_files)), desc="INT8 Calibration", total=len(image_files)):
            # print(input_value)

            # img = Image.open(image_files[input_value]).resize(input_size, Image.NEAREST)
            img = cv2.imread(image_files[input_value])

            hs, ws = img.shape[:2]
            wt, ht = input_size

            if wt / ht > hs / ws:
                crop_size = int(hs * (wt / ht)), hs
            else:
                crop_size = ws, int(ws * (ht / wt))

            img = crop_center(img, crop_size)
            img = cv2.resize(img, (wt, ht))

            # cv2.imwrite(f"calib_log/calib{input_value}.jpg", img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)

            yield [tf.convert_to_tensor(img[None])]

    return data_gen
