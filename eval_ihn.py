import os
import gc
import yaml
import utils
import torch
import models
import argparse
import datasets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


torch.manual_seed(0)


def prepare_eval(config):
    spec = config.get('eval_dataset')
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    sv_file = torch.load(config['resume_align'])
    H_model = models.IHN().cuda()
    H_model.load_state_dict(sv_file['model']['sd'])

    return loader, H_model


def eval(loader, model, args):
    model.eval()

    failures = 0

    tot_rmse, tot_mace = 0, 0
    tot_psnr, tot_ssim = 0, 0

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)

    for b_id in pbar:
        batch = next(loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp_tgt = batch['inp_ref'].permute(0,3,1,2)
        inp_src = batch['inp_tgt'].permute(0,3,1,2)

        inp_tgt_ = inp_tgt * 255
        inp_src_ = inp_src * 255

        mask = torch.ones_like(inp_src).cuda()
        b, c, h, w = inp_tgt.shape

        four_pred, _ = model(inp_src_, inp_tgt_, iters_lev0=6, iters_lev1=3, test_mode=True)
        shift = four_pred.reshape(b, 2, -1).permute(0, 2, 1)

        shape = (128, 128)
        H, w_max, w_min, h_max, h_min = utils.get_H(shift, shape)
        H = utils.compens_H(H, size=shape)

        img_h = torch.ceil(h_max - h_min).int().item()
        img_w = torch.ceil(w_max - w_min).int().item()

        h_max = h_max.item(); h_min = h_min.item()
        w_max = w_max.item(); w_min = w_min.item()

        src_w_ovl = utils.STN(inp_src, torch.inverse(H))
        mask_ovl = utils.STN(mask, torch.inverse(H)).round().bool()
        tgt_ovl = inp_tgt * mask_ovl

        src_samples = src_w_ovl[mask_ovl].cpu().numpy()
        tgt_samples = tgt_ovl[mask_ovl].cpu().numpy()

        if len(src_samples) == 0:
            failures += 1
            continue

        psnr = compare_psnr(tgt_samples, src_samples, data_range=1.)
        ssim = compare_ssim(tgt_samples * 255, src_samples * 255, data_range=255.)

        tot_psnr += psnr
        tot_ssim += ssim

        pbar.set_description_str(
        desc="PSNR:{:.4f}, SSIM:{:.4f}, Failures:{}".format(
            tot_psnr/(b_id+1), tot_ssim/(b_id+1), failures), refresh=True)


def main(config, args):
    loader, model = prepare_eval(config)

    with torch.no_grad():
        eval(loader, model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    main(config, args)
