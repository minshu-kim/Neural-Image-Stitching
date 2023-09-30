import os
import gc
import yaml
import utils

import models
import datasets
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from srwarp import transform
from torch.optim.lr_scheduler import ExponentialLR


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=16, pin_memory=True)
    return loader


def prepare_training(resume=False):
    if resume:
        sv_file = torch.load(config['resume_stitching'])
        model = models.make(sv_file['model'], load_sd=True).cuda()

        H_model = models.IHN().cuda()
        sv_file_align = torch.load(config['resume_align'])
        H_model.load_state_dict(sv_file_align['model']['sd'])

        params = model.imnet.parameters()
        optimizer = utils.make_optimizer(params, sv_file['optimizer'], load_sd=True)

        epoch_start = sv_file['epoch'] + 1
        lr_scheduler = ExponentialLR(optimizer, gamma=0.96)

    else:
        sv_file = torch.load(config['resume_stitching'])
        model = models.make(sv_file['model'], load_sd=True).cuda()

        H_model = models.IHN().cuda()
        sv_file = torch.load(config['resume_align'])
        H_model.load_state_dict(sv_file['model']['sd'])

        params = model.imnet.parameters()
        optimizer = utils.make_optimizer(params, config['optimizer'])

        epoch_start = 1
        lr_scheduler = ExponentialLR(optimizer, gamma=0.96)

    sv_file = torch.load(config['resume_stitching_eval'])
    model_for_eval = models.make(sv_file['model'], load_sd=True).cuda()

    for param in model_for_eval.parameters():
        param.requires_grad = False

    evaluator = models.evaluator.SeamEvaluator()
    n_params = utils.compute_num_params(model, text=False) + utils.compute_num_params(H_model, text=False)

    log('model: params={}'.format(str(n_params)))

    return model, H_model, optimizer, epoch_start, lr_scheduler, evaluator, model_for_eval


def get_ingredient(ref, tgt, H_model):
    b, c, h, w = ref.shape

    if h != 128 or w != 128:
        inp_ref = F.interpolate(ref, size=(128,128), mode='bilinear')
        inp_tgt = F.interpolate(tgt, size=(128,128), mode='bilinear')
    else:
        inp_ref = ref.contiguous()
        inp_tgt = tgt.contiguous()

    four_pred, _ = H_model(inp_tgt * 255, inp_ref * 255, iters_lev0=6, iters_lev1=3, test_mode=True)
    shift = four_pred.reshape(b, 2, -1).permute(0, 2, 1)

    shape = (h, w)
    H_tgt2ref, w_max, w_min, h_max, h_min = utils.get_H(shift * w/128, shape)

    img_h = torch.ceil(h_max - h_min).int().item()
    img_w = torch.ceil(w_max - w_min).int().item()
    sizes = (img_h, img_w)

    h_max.clamp_(0, h*2); w_max.clamp_(0, w*2)
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
        cell1.contiguous(), *sizes, *ref.shape[-2:], H_tgt2ref.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1).cuda()

    coord2 = coord.clone()
    ref_grid, ref_mask = utils.gridy2gridx_homography(
        coord2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    )

    cell2 = cell.clone()
    ref_cell = utils.celly2cellx_homography(
        cell2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1).cuda()

    ref_grid = ref_grid.unsqueeze(0).repeat(b,1,1).cuda()
    tgt_grid = tgt_grid.unsqueeze(0).repeat(b,1,1).cuda()
    ref_mask = ref_mask.reshape(b,1,img_h,img_w)
    tgt_mask = tgt_mask.reshape(b,1,img_h,img_w)

    return tgt_grid, tgt_cell, tgt_mask, ref_grid, ref_cell, ref_mask, sizes


def train(train_loader, model, H_model, evaluator, optimizer, epoch, model_for_eval, args):
    model.train()
    H_model.train()
    model_for_eval.eval()

    tot_loss = 0
    l1_loss = nn.L1Loss()

    train_loader = iter(train_loader)
    pbar = tqdm(range(len(train_loader)), smoothing=0.9)

    for b_id in pbar:
        batch = next(train_loader)

        for k, v in batch.items():
            batch[k] = v.cuda()

        ref = batch['inp_ref']
        tgt = batch['inp_tgt']

        b, c, h, w = ref.shape

        tgt_grid, tgt_cell, tgt_mask, ref_grid, ref_cell, ref_mask, sizes = get_ingredient(ref, tgt, H_model)
        stit_grid = utils.to_pixel_samples(None, sizes)

        ref = (ref - 0.5) * 2
        tgt = (tgt - 0.5) * 2

        seam_ref = evaluator.get_seam_mask(ref_mask, tgt_mask) * ref_mask
        seam_tgt = evaluator.get_seam_mask(tgt_mask, ref_mask) * tgt_mask

        num_samples = 48**2
        tgt_q = np.random.choice(
                np.nonzero(seam_tgt.flatten(2)[0,0].cpu())[:, 0],
                min(num_samples, int(torch.sum(seam_tgt).item())), replace=False)

        ref_q = np.random.choice(
                np.nonzero(seam_ref.flatten(2)[0,0].cpu())[:, 0],
                min(num_samples, int(torch.sum(seam_ref).item())), replace=False)

        if len(tgt_q) == 0 or len(ref_q) == 0:
            pred = None; loss = None
            torch.cuda.empty_cache()
            gc.collect()
            continue

        stit_grid = stit_grid.unsqueeze(0).repeat(b,1,1).cuda()
        queries = np.concatenate((ref_q, tgt_q), axis=0)

        stit_grid_s = stit_grid[:, queries].cuda()
        stit_grid_s_eval_tgt = stit_grid[:, tgt_q].clone().contiguous().cuda()
        stit_grid_s_eval_ref = stit_grid[:, ref_q].clone().contiguous().cuda()

        pred = model(
            ref, ref_grid, ref_cell, ref_mask,
            tgt, tgt_grid, tgt_cell, tgt_mask,
            stit_grid_s, sizes
        )

        pred_ref = pred[:, :ref_q.shape[0]]
        pred_tgt = pred[:, ref_q.shape[0]:]

        with torch.no_grad():
            eval_ref = model_for_eval(
                ref, ref_grid, ref_cell, ref_mask,
                ref, ref_grid, ref_cell, ref_mask,
                stit_grid_s_eval_ref, sizes, eval_mode=True
            ).clamp(-1,1)

            eval_tgt = model_for_eval(
                tgt, tgt_grid, tgt_cell, tgt_mask,
                tgt, tgt_grid, tgt_cell, tgt_mask,
                stit_grid_s_eval_tgt, sizes, eval_mode=True
            ).clamp(-1,1)

        loss = l1_loss(eval_tgt, pred_tgt) + l1_loss(eval_ref, pred_ref)
        tot_loss += loss.item()

        pbar.set_description_str(desc="[Train] Epoch:{}/{}".format(epoch, config['epoch_max']), refresh=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None

    return tot_loss / (b_id+1)


def main(config_, save_path, args):
    global config, log
    config = config_
    log, _ = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    model, H_model, optimizer, epoch_start, lr_scheduler, evaluator, model_for_eval = prepare_training(resume=args.resume)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    timer = utils.Timer()

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')

    sv_file = torch.load(config['resume_stitching'])

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        train_loss = train(train_loader, model, H_model, evaluator, optimizer, epoch, model_for_eval, args)
        lr_scheduler.step()

        log_info.append('train loss: {:.4f}'.format(train_loss))

        if n_gpus > 1:
            sv_model = model.module
        else:
            sv_model = model

        model_spec = sv_file['model']
        model_spec['sd'] = sv_model.state_dict()

        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
        log(', '.join(log_info))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_path = os.path.join('./save', '_' + args.config.split('/')[-1][:-len('.yaml')])
    main(config, save_path, args)

