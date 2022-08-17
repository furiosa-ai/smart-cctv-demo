# Setup
Install packages
```bash
sudo apt install libeigen3-dev  # Ubuntu
brew install eigen  # Mac
```

Setup environment
```bash
git clone git@github.com:furiosa-ai/yolov5.git
cd yolov5
conda activate <env>
pip install -r requirements.txt
export PYTHONPATH=.
./build.sh
```

# MOT

## Files

1. Download weights (person detection) from [here](https://drive.google.com/drive/folders/1NP7qlT3hbdVSqK8MVGIH2iH1MNz2x_Bs?usp=sharing) and extract to runs/train/bytetrack_mot20_5data

2. Download test videos from [here](https://drive.google.com/file/d/1aBmB1JqiyJMUj0USbgaEt-owBKpCe2B9/view?usp=sharing) and extract to datasets/MOT20/vid/

## Run

```bash
python detect.py \
  --source datasets/MOT20/vid/MOT20-02-640.mp4 \
  --cfg models/yolov5m_warboy.yaml \
  --weights runs/train/bytetrack_mot20_5data/weights/best.pt \
  --view-img \
  --img-size 360 640 \
  --track
```

# Multi Camera MOT (MCMOT) Demo

## Files

1. Download weights (person detection) from [here](https://drive.google.com/drive/folders/1NP7qlT3hbdVSqK8MVGIH2iH1MNz2x_Bs?usp=sharing) and extract to runs/train/bytetrack_mot20_5data
2. Download test videos from [here](https://drive.google.com/file/d/1aBmB1JqiyJMUj0USbgaEt-owBKpCe2B9/view?usp=sharing) and extract to datasets/MOT20/vid/
3. Download calibration data (quantization) from [here](https://drive.google.com/drive/folders/1JT-FZR1nl4BOhopAuUOEuLvpiy7o3zWb?usp=sharing) and extract to datasets/calib/coco


## Run demo

<details open>
<summary>Run and display</summary>

```bash
python run/mcmot_demo.py --cfg cfg/mot_fur15_pipeline_mot_only_3plot_mot20.yaml \
    --device cpu  # optionally override cfg (cpu, cuda, furiosa)
```
</details>


# YOLOv5

[YOLOv5 README](README_yolo.md)