import os
import gc
import cv2

import yaml
import utils
import models
import datasets
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from srwarp import transform

from torchvision import transforms
from torch.utils.data import DataLoader

c1 = [0, 0]; c2 = [0, 0]
finder = cv2.detail.SeamFinder.createDefault(2)

def seam_finder(ref, tgt, ref_m, tgt_m):
    ref_ = ref.mean(dim=1, keepdim=True)
    ref_ -= ref_.min()
    ref_ /= ref_.max()

    tgt_ = tgt.mean(dim=1, keepdim=True)
    tgt_ -= tgt_.min()
    tgt_ /= tgt_.max()

    ref_ = (ref_.cpu().numpy() * 255).astype(np.uint8)
    tgt_ = (tgt_.cpu().numpy() * 255).astype(np.uint8)
    ref_m = (ref_m[0,0,:,:].cpu().numpy() * 255).astype(np.uint8)
    tgt_m = (tgt_m[0,0,:,:].cpu().numpy() * 255).astype(np.uint8)

    inp = np.concatenate([ref_, tgt_], axis=0).transpose(0,2,3,1)
    inp = np.repeat(inp, 3, -1)

    masks = np.stack([ref_m, tgt_m], axis=0)[..., None]
    corners = np.stack([c1, c2], axis=0).astype(np.uint8)

    ref_m, tgt_m = finder.find(inp, corners, masks)
    ref *= torch.Tensor(cv2.UMat.get(ref_m).reshape(1,1,*ref.shape[-2:])/255)
    tgt *= torch.Tensor(cv2.UMat.get(tgt_m).reshape(1,1,*tgt.shape[-2:])/255)

    stit_rep = ref + tgt

    return stit_rep


def batched_predict(model, ref, ref_grid, ref_cell, ref_mask,
                    tgt, tgt_grid, tgt_cell, tgt_mask,
                    stit_grid, sizes, bsize, seam_cut=False):

    fea_ref, fea_ref_grid, ref_coef, ref_freq = model.gen_feat(ref)
    fea_ref_w = model.NeuralWarping(
        ref, fea_ref, fea_ref_grid,
        ref_freq, ref_coef, ref_grid, ref_cell, sizes
    ).cpu()

    fea_tgt, fea_tgt_grid, tgt_coef, tgt_freq = model.gen_feat(tgt)
    fea_tgt_w = model.NeuralWarping(
        tgt, fea_tgt, fea_tgt_grid,
        tgt_freq, tgt_coef, tgt_grid, tgt_cell, sizes
    ).cpu()

    if seam_cut:
        fea = seam_finder(fea_ref_w, fea_tgt_w, ref_mask, tgt_mask).repeat(1,2,1,1).cuda()

    else:
        fea_ref_w *= ref_mask
        fea_tgt_w *= tgt_mask
        fea = torch.cat([fea_ref_w, fea_tgt_w], dim=1).cuda()

    stit_rep = model.gen_feat_for_blender(fea)

    ql = 0
    preds = []
    n = ref_grid.shape[1]

    while ql < n:
        qr = min(ql + bsize, n)
        pred = model.query_rgb(stit_rep, stit_grid[:, ql: qr, :])
        preds.append(pred)
        ql = qr

    pred = torch.cat(preds, dim=1)

    return pred


def prepare_ingredient(model, inp_tgt, inp_ref, tgt, ref):
    b, c, h, w = tgt.shape

    four_pred, _ = model(inp_tgt, inp_ref, iters_lev0=6, iters_lev1=3, test_mode=True)
    shift = four_pred.reshape(b, 2, -1).permute(0, 2, 1)

    shape = tgt.shape[-2:]
    H_tgt2ref, w_max, w_min, h_max, h_min = utils.get_H(shift * w/128, shape)

    img_h = torch.ceil(h_max - h_min).int().item()
    img_w = torch.ceil(w_max - w_min).int().item()
    sizes = (img_h, img_w)

    h_max = h_max.item(); h_min = h_min.item()
    w_max = w_max.item(); w_min = w_min.item()

    eye = torch.eye(3).double()
    T = utils.get_translation(h_min, w_min)

    H_tgt2ref = H_tgt2ref[0].double().cpu()
    H_tgt2ref = T @ H_tgt2ref

    eye, _, _ = transform.compensate_matrix(ref, eye)
    eye = T @ eye

    coord = utils.to_pixel_samples(None, sizes=sizes)
    cell = utils.make_cell(coord, None, sizes=sizes).cuda()
    coord = coord.cuda()

    coord1 = coord.clone()
    tgt_grid, tgt_mask = utils.gridy2gridx_homography(
        coord1.contiguous(), *sizes, *ref.shape[-2:], H_tgt2ref.cuda(), cpu=False
    )

    cell1 = cell.clone()
    tgt_cell = utils.celly2cellx_homography(
        cell1.contiguous(), *sizes, *tgt.shape[-2:], H_tgt2ref.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1)

    coord2 = coord.clone()
    ref_grid, ref_mask = utils.gridy2gridx_homography(
        coord2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    )

    cell2 = cell.clone()
    ref_cell = utils.celly2cellx_homography(
        cell2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1)

    stit_grid = utils.to_pixel_samples(None, sizes).cuda()
    stit_mask = (tgt_mask + ref_mask).clamp(0,1)

    ref_grid = ref_grid.unsqueeze(0).repeat(b,1,1)
    tgt_grid = tgt_grid.unsqueeze(0).repeat(b,1,1)
    stit_grid = stit_grid.unsqueeze(0).repeat(b,1,1)

    return tgt_grid, tgt_cell, tgt_mask, ref_grid, ref_cell, ref_mask, stit_grid, stit_mask, sizes


def prepare_validation(config):
    sv_file = torch.load("pretrained/NIS_blending.pth")
    model = models.make(sv_file['model'], load_sd=True).cuda()

    H_model = models.IHN().cuda()
    sv_file = torch.load("pretrained/ihn.pth")
    H_model.load_state_dict(sv_file['model']['sd'])

    return model, H_model


def valid(model, H_model, ref, tgt):
    model.eval()
    H_model.eval()

    tot_psnr = 0
    failures = []

    os.makedirs("visualization/", exist_ok=True)

    b, c, h, w = ref.shape

    # In the case of GPU Out-of-memory
    # ref = F.interpolate(ref, size=(1024, 1024), mode='bilinear')
    # tgt = F.interpolate(tgt, size=(1024, 1024), mode='bilinear')

    if ref.shape[-2:] != tgt.shape[-2:]:
        tgt = F.interpolate(tgt, size=(h, w), mode='bilinear')

    if h != 128 or w != 128:
        inp_ref = F.interpolate(ref, size=(128,128), mode='bilinear') * 255
        inp_tgt = F.interpolate(tgt, size=(128,128), mode='bilinear') * 255

    else:
        inp_ref = ref * 255
        inp_tgt = tgt * 255

    tgt_grid, tgt_cell, tgt_mask, \
    ref_grid, ref_cell, ref_mask, \
    stit_grid, stit_mask, sizes = prepare_ingredient(H_model, inp_tgt, inp_ref, tgt, ref)

    ref = (ref - 0.5) * 2
    tgt = (tgt - 0.5) * 2

    ref_mask = ref_mask.reshape(b,1,*sizes)
    tgt_mask = tgt_mask.reshape(b,1,*sizes)

    pred = batched_predict(
        model, ref, ref_grid, ref_cell, ref_mask,
        tgt, tgt_grid, tgt_cell, tgt_mask,
        stit_grid, sizes, config['eval_bsize'], seam_cut=True
    )
    pred = pred.permute(0, 2, 1).reshape(b, c, *sizes)
    pred = ((pred + 1)/2).clamp(0,1) * stit_mask.reshape(b, 1, *sizes)

    transforms.ToPILImage()(pred[0]).save('results.png')


def main(config_, args):
    global config
    config = config_
    model, H_model = prepare_validation(config)

    ref = transforms.ToTensor()(
        Image.open(args.ref).convert('RGB')
    ).cuda().unsqueeze(0)

    tgt = transforms.ToTensor()(
        Image.open(args.tgt).convert('RGB')
    ).cuda().unsqueeze(0)

    with torch.no_grad():
        valid(model, H_model, ref, tgt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--ref', default='ref.jpg')
    parser.add_argument('--tgt', default='tgt.jpg')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    main(config, args)
