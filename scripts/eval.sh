CUDA_VISIBLE_DEVICES=$1 python eval_enhancing.py --config=configs/test/NIS_enhancing.yaml
CUDA_VISIBLE_DEVICES=$1 python eval_blending.py --config=configs/test/NIS_blending.yaml
