# Implicit Neural Image Stitching

This repository contains the official implementation of NIS, 24' WACV: https://arxiv.org/abs/2309.01409

## Requirement
1) Python packages
```
conda env create --file environment.yaml
conda activate nis
```
2) [pysrwarp](https://github.com/sanghyun-son/srwarp)
: Follow guidelines in the repository for more details.
```
git clone https://github.com/sanghyun-son/pysrwarp
cd pysrwarp
make
```

## Train & Evaluation
```
bash scripts/train.sh 0
bash scripts/eval.sh 0
```

## Pretrained Models
You can download below models on this [link](https://drive.google.com/file/d/1sdfquwxhKLq2aBGGdtiu8_SM-g-aDUtM/view?usp=share_link).
1. NIS_enhancing.pth: Pretrained on Enhanced Stitching (Stage 1),
2. NIS_blending.pth: Pretrained on Enhanced & Blended Stitching (Stage 1 & 2),
3. ihn.pth: Our reproduced Homography Estimator used in the second stage training.

## Acknowlegment
This work is mainly based on [LTEW](https://github.com/jaewon-lee-b/ltew) and [IHN](https://github.com/imdumpl78/IHN), we thank the authors for the contribution.
