# Validation


## Files
- Download weights from [here](https://drive.google.com/file/d/1MRi7ERTlJA9y4kre6p_cBDxm1JkEBZQr/view?usp=sharing) and extract to runs/train
- Download BDD validation dataset from [here](https://drive.google.com/file/d/1Q7AhMkmM3BKeVVa_ilsbaFqOK9GROn6Q/view?usp=sharing) and extract to datasets/


## Validate

<details open>
<summary>Yolov5m (original)</summary>

```bash
# CUDA
python val_furiosa.py \
  --cfg yolov5m.yaml \
  --weights runs/train/m_bdd/weights/best.pt \
  --data bdd100k_val.yaml \
  --imgsz 640 \
  --batch-size 1 \
  --device cuda
```

```bash
# Furiosa MinMax calibration with 10 images (from BDD)
python val_furiosa.py \
  --cfg yolov5m.yaml \
  --weights runs/train/m_bdd/weights/best.pt \
  --data bdd100k_val.yaml \
  --imgsz 640 \
  --batch-size 1 \
  --device furiosa \
  --calib_mode minmax \
  --calib_count 10
```

```bash
# Furiosa entropy calibration with 1000 images (from BDD)
python val_furiosa.py \
  --cfg yolov5m.yaml \
  --weights runs/train/m_bdd/weights/best.pt \
  --data bdd100k_val.yaml \
  --imgsz 640 \
  --batch-size 1 \
  --device furiosa \
  --calib_mode entropy \
  --calib_count 1000
```

</details>


<details open>
<summary>Yolov5m (concat -> add)</summary>

```bash
# CUDA
python val_furiosa.py \
  --cfg yolov5m_warboy.yaml \
  --weights runs/train/mwarboy_bdd/weights/best.pt \
  --data bdd100k_val.yaml \
  --imgsz 640 \
  --batch-size 1 \
  --device cuda
```

```bash
# Furiosa MinMax calibration with 10 images (from BDD)
python val_furiosa.py \
  --cfg yolov5m_warboy.yaml \
  --weights runs/train/mwarboy_bdd/weights/best.pt \
  --data bdd100k_val.yaml \
  --imgsz 640 \
  --batch-size 1 \
  --device furiosa \
  --calib_mode minmax \
  --calib_count 10
```

```bash
# Furiosa entropy calibration with 1000 images (from BDD)
python val_furiosa.py \
  --cfg yolov5m_warboy.yaml \
  --weights runs/train/mwarboy_bdd/weights/best.pt \
  --data bdd100k_val.yaml \
  --imgsz 640 \
  --batch-size 1 \
  --device furiosa \
  --calib_mode entropy \
  --calib_count 1000
```

</details>