## Setup

```bash
sudo apt install cmake
sudo apt install \
  furiosa-libhal-warboy=0.9.0-2+nightly-220810 \
  libonnxruntime=1.11.1-2 \
  furiosa-libnux=0.8.0-2+nightly-220810 \
  furiosa-libcompiler=0.8.0-2+nightly-220803
conda env create -n "cctv" python=3.8
conda activate cctv
conda install cython
conda env update --file environment.yml
bash build.sh
```

## Datasets

- Download CrowdHuman from [here](https://drive.google.com/file/d/18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO/view) and extract to "data/CrowdHuman/"
- Download PRW from [here](https://anu365-my.sharepoint.com/personal/u1064892_anu_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fu1064892%5Fanu%5Fedu%5Fau%2FDocuments%2FPRW%2Dv16%2E04%2E20%2Ezip&parent=%2Fpersonal%2Fu1064892%5Fanu%5Fedu%5Fau%2FDocuments&ga=1) and extract to "data/"
- Download WIDERFACE from [here](https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view) and extract to "data/"
- Download MSM1-RetinaFace from [here](https://drive.google.com/file/d/1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy/view) and extract to "data/"
- market1501 dataset will be downloaded automatically at progam execution

## Test files (face recognition)

- Download [video](https://www.youtube.com/watch?v=PmvsAi89BDM) and move to "data/tom_cruise_test.mp4"
- Download [query](https://www.biography.com/.image/ar_1:1%2Cc_fill%2Ccs_srgb%2Cg_face%2Cq_auto:good%2Cw_300/MTc5ODc1NTM4NjMyOTc2Mzcz/gettyimages-693134468.jpg) and move to "data/tom_cruise_profile1.jpeg"

## Weights

- Download reid checkpoint from [here](https://drive.google.com/file/d/1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV/view) and move to weights/
- Download remaining checkpoints from [here](https://drive.google.com/file/d/11cpf9_HC-oK_wFBdVEoUVENakDpwzNuZ/view?usp=sharing) and extract to weights/

## Final directory structure

```bash
smart-cctv-demo$ tree -L 2 data weights
data
├── CrowdHuman
│   └── Images
├── PRW-v16.04.20
│   ├── ID_test.mat
│   ├── ID_train.mat
│   ├── annotations
│   ├── frame_test.mat
│   ├── frame_train.mat
│   ├── frames
│   ├── generate_query.m
│   ├── query_box
│   ├── query_info.txt
│   └── readme.txt
├── WIDER_val
│   └── images
├── market1501
│   ├── Market-1501-v15.09.15
│   └── Market-1501-v15.09.15.zip
├── ms1m-retinaface-t1
│   ├── agedb_30.bin
│   ├── cfp_fp.bin
│   ├── lfw.bin
│   ├── property
│   ├── train.idx
│   ├── train.lst
│   └── train.rec
├── tom_cruise_profile1.jpeg
└── tom_cruise_test.mp4
weights
├── ms1mv3_r50_leakyrelu_1.pt
├── resnet50_market_xent.pth.tar
├── yolov5_face_relu.pt
└── yolov5m_warboy.pt
```

## Query demo

```bash
# Optionally clear cache
rm *.onnx  # f32 model cache
rm *.dfg  # i8 model cache
rm galleries/*  # gallery cache
rm -rf results/*  # delete previous results

# Face recognition (results will be written to "results/face")
cd insightface
python recognition/arcface_torch/query_video.py \
  --gallery ../data/tom_cruise_test.mp4 \
  --query ../data/tom_cruise_profile1.jpeg \
  --device furiosa

# ReId (results will be written to "results/reid")
cd torchreid
python query_video.py \
  --gallery ../data/PRW-v16.04.20/frames \
  --query ../data/PRW-v16.04.20/query_box/479_c1s3_016471.jpg \
  --device furiosa
```

## GUI

- Local inference

  ```bash
  python demo.py
  ```

- Server inference

  Add to ~/.ssh/config

  ```
  Host vis
    HostName 210.109.63.237
    User ubuntu
    IdentityFile ~/.ssh/vision-intelligence.pem
    IdentitiesOnly yes
  ```

  Run
  ```bash
  python demo.py --server cfg/server_vis.yaml
  ```

## Demo


<details><summary>Face recognition</summary>

  ![face recognition](doc/face_reg.gif)
</details>


<details><summary>Person recognition</summary>

  ![person detection](doc/reid.gif)
</details>

