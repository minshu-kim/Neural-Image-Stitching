# Implicit Neural Image Stitching

This repository contains the official implementation of [NIS](https://arxiv.org/abs/2309.01409), 24' WACV.

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

## Acknowlegment
This work is mainly based on [LTEW](https://github.com/jaewon-lee-b/ltew) and [IHN](https://github.com/imdumpl78/IHN), we thank the authors for the contribution.
