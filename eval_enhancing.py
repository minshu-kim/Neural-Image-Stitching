import argparse
import os
import yaml

import utils
import models
import datasets

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

torch.manual_seed(2022)

def batched_predict(model, ref, ref_grid, ref_cell, ref_mask,
                    tgt, tgt_grid, tgt_cell, tgt_mask,
                    stit_grid, sizes, bsize):

    with torch.no_grad():
        fea_ref, fea_ref_grid, ref_coef, ref_freq = model.gen_feat(ref)
        fea_ref_w = model.NeuralWarping(
            ref, fea_ref, fea_ref_grid,
            ref_freq, ref_coef, ref_grid, ref_cell, sizes
        )

        fea_tgt, fea_tgt_grid, tgt_coef, tgt_freq = model.gen_feat(tgt)
        fea_tgt_w = model.NeuralWarping(
            tgt, fea_tgt, fea_tgt_grid,
            tgt_freq, tgt_coef, tgt_grid, tgt_cell, sizes
        )

        fea = torch.cat([fea_ref_w * ref_mask, fea_tgt_w * tgt_mask], dim=1)
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


def get_ingredients(img):
    b, c, h, w = img.shape

    try:
        ref, tgt, H_tgt2ref, eye, gt_stit, mask_stit, sizes = utils.random_warp(img, box_size=128, ovl_rate=0.75, offset_ratio=0.20)
    except:
        return None

    max_wh, min_wh = sizes
    (w_max, h_max), (w_min, h_min) = max_wh, min_wh

    img_h = torch.ceil(h_max - h_min).int().item()
    img_w = torch.ceil(w_max - w_min).int().item()
    h_max = h_max.item(); h_min = h_min.item(); w_max = w_max.item(); w_min = w_min.item()

    sizes = (img_h, img_w)

    coord = utils.to_pixel_samples(None, sizes=sizes)
    cell = utils.make_cell(coord, None, sizes=sizes).cuda()
    coord = coord.cuda()

    coord1 = coord.clone()
    tgt_grid, tgt_mask = utils.gridy2gridx_homography(
        coord1.contiguous(), *sizes, *ref.shape[-2:], H_tgt2ref.cuda(), cpu=False
    )

    cell1 = cell.clone()
    tgt_cell = utils.celly2cellx_homography(
        cell1.contiguous(), *sizes, *ref.shape[-2:], H_tgt2ref.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1)

    coord2 = coord.clone()
    ref_grid, ref_mask = utils.gridy2gridx_homography(
        coord2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    )

    cell2 = cell.clone()
    ref_cell = utils.celly2cellx_homography(
        cell2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1)

    stit_grid = utils.to_pixel_samples(None, sizes).cuda().unsqueeze(0).repeat(b,1,1)
    ref_grid = ref_grid.unsqueeze(0).repeat(b,1,1)
    tgt_grid = tgt_grid.unsqueeze(0).repeat(b,1,1)

    ref_mask = ref_mask.reshape(b,1,img_h,img_w)
    tgt_mask = tgt_mask.reshape(b,1,img_h,img_w)
    stit_mask = (ref_mask + tgt_mask).clamp(0,1)

    return tgt, tgt_grid, tgt_cell, tgt_mask, ref, ref_grid, ref_cell, ref_mask, stit_grid, stit_mask, gt_stit, sizes


def prepare(config):
    spec = config.get('dataset')
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('dataset size: {}'.format(len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    sv_file = torch.load(config['resume'])
    model = models.make(sv_file['model'], load_sd=True).cuda()

    log('model params: {}'.format(utils.compute_num_params(model, text=True)))

    return loader, model


def valid(loader, model):
    model.eval()

    count = 0
    tot_psnr = 0
    tot_ssim = 0

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)

    for b_id in pbar:
        batch = next(loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        img = (batch['img'] - 0.5) * 2
        b, _, h, w = img.shape

        try:
            tgt, tgt_grid, tgt_cell, tgt_mask, \
            ref, ref_grid, ref_cell, ref_mask, \
            stit_grid, stit_mask, gt_stit, sizes = get_ingredients(img)
        except:
            count += 1
            continue

        pred = batched_predict(
            model, ref, ref_grid, ref_cell, ref_mask,
            tgt, tgt_grid, tgt_cell, tgt_mask,
            stit_grid, sizes, config['eval_bsize']
        )

        pred = pred.permute(0,2,1).reshape(b,3,*sizes)
        pred = (pred * stit_mask).cuda()
        gt = (gt_stit * stit_mask).cuda()

        stit_mask = stit_mask.repeat(1,3,1,1).bool()
        pred_samples = pred[stit_mask].cpu().numpy()
        gt_samples = gt[stit_mask].cpu().numpy()

        tot_psnr += compare_psnr(pred_samples, gt_samples, data_range=2.)
        tot_ssim += compare_ssim(gt_samples, pred_samples, data_range=2.)

        pbar.set_description_str(desc="[Valid] PSNR: {:.4f}, SSIM: {:.4f}".format(
            tot_psnr/(b_id-count+1), tot_ssim/(b_id-count+1)), refresh=True)

        pred = None

    return tot_psnr/(b_id-count+1)


def main(config_, save_path, args):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    loader, model = prepare(config)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    with torch.no_grad():
        valid_psnr = valid(loader, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_path = os.path.join('./save', '_' + args.config.split('/')[-1][:-len('.yaml')])
    main(config, save_path, args)
