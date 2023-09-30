CUDA_VISIBLE_DEVICES=$1 python train_enhancing.py --config=configs/train/NIS_enhancing.yaml
CUDA_VISIBLE_DEVICES=$1 python train_blending.py --config=configs/train/NIS_blending.yaml
