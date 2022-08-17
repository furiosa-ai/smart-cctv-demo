## Setup

```bash
sudo apt install cmake
conda env create -n "cctv" python=3.8
conda activate cctv
conda install cython
conda env update --file environment.yml
bash build.sh
```

## Datasets

- Download CrowdHuman from [here](https://drive.google.com/file/d/18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO/view) and extract to "data/CrowdHuman"
- Download PRW from [here](https://anu365-my.sharepoint.com/personal/u1064892_anu_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fu1064892%5Fanu%5Fedu%5Fau%2FDocuments%2FPRW%2Dv16%2E04%2E20%2Ezip&parent=%2Fpersonal%2Fu1064892%5Fanu%5Fedu%5Fau%2FDocuments&ga=1) and extract to "data/PRW-v16.04.20"
- Download WIDERFACE from [here](https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view) and extract to "data/WIDERFACE"
- Download MSM1-RetinaFace from [here](https://drive.google.com/file/d/1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy/view) and extract to "data/WIDERFACE"
- market1501 dataset will be downloaded automatically at progam execution

## Test files (face recognition)

- Download [video](https://www.youtube.com/watch?v=PmvsAi89BDM) and move to "data/tom_cruise_test.mp4"
- Download [query](https://www.biography.com/.image/ar_1:1%2Cc_fill%2Ccs_srgb%2Cg_face%2Cq_auto:good%2Cw_300/MTc5ODc1NTM4NjMyOTc2Mzcz/gettyimages-693134468.jpg) and move to "data/tom_cruise_profile1.jpeg"

## Weights

- Download reid checkpoint from [here](https://drive.google.com/file/d/1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV/view) and move to weights/
- Download remaining checkpoints from [here](https://drive.google.com/file/d/11cpf9_HC-oK_wFBdVEoUVENakDpwzNuZ/view?usp=sharing) and extract to weights/

## Final directory structure

## Query demo

```bash
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