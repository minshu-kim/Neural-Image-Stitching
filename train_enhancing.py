import os
import gc
import yaml
import torch
import argparse
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import utils
import models
import datasets
import numpy as np

from torchvision import transforms
from srwarp import transform, warp, crop


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


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    valid_loader = make_data_loader(config.get('valid_dataset'), tag='valid')
    return train_loader, valid_loader


def prepare_training(resume=False):
    if resume:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(model.parameters(), sv_file['optimizer'], load_sd=True)

        epoch_start = sv_file['epoch'] + 1
        lr_scheduler = ExponentialLR(optimizer, gamma=0.98)

    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])

        for param in model.blender.parameters():
            param.requires_grad = False

        epoch_start = 1
        lr_scheduler = ExponentialLR(optimizer, gamma=0.98)

    log('Model Params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def get_ingredient(ref, mask_stit, H_tgt2ref, eye, sizes):
    b, c, h, w = ref.shape

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
    tgt_mask = tgt_mask.reshape(1,1,*sizes)

    coord2 = coord.clone()
    ref_grid, ref_mask = utils.gridy2gridx_homography(
        coord2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    )
    ref_mask = ref_mask.reshape(1,1,*sizes)

    cell2 = cell.clone()
    ref_cell = utils.celly2cellx_homography(
        cell2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1)

    stit_grid = utils.to_pixel_samples(None, sizes)
    stit_mask = ((tgt_mask + ref_mask)/2).clamp(0,1)

    num_samples = 48**2
    queries = np.random.choice(
            np.nonzero(mask_stit.flatten(2)[0,0].cpu())[:, 0], min(num_samples, int(torch.sum(mask_stit).item())
        ), replace=False
    )

    ref_grid = ref_grid.unsqueeze(0).repeat(b,1,1)
    tgt_grid = tgt_grid.unsqueeze(0).repeat(b,1,1)

    stit_grid = stit_grid.unsqueeze(0).repeat(b,1,1)
    stit_grid_s = stit_grid[:, queries].cuda()

    return tgt_grid, tgt_cell, tgt_mask, ref_grid, ref_cell, ref_mask, stit_grid_s, queries


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_fn = nn.L1Loss()

    tot_loss = 0
    tot_psnr = 0

    train_loader = iter(train_loader)
    print('[Scheduler] lr: {}'.format(optimizer.param_groups[0]['lr']))

    pbar = tqdm(range(len(train_loader)), smoothing=0.9)

    for b_id in pbar:
        batch = next(train_loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        img = (batch['inp'] - 0.5) * 2
        b, _, h, w = img.shape

        try:
            ref, tgt, H_tgt2ref, eye, \
            gt_stit, mask_stit, sizes = utils.random_warp(img, box_size=48, ovl_rate=0.75, offset_ratio=0.25)
        except:
            continue

        max_wh, min_wh = sizes
        (w_max, h_max), (w_min, h_min) = max_wh, min_wh

        img_h = torch.ceil(h_max - h_min).int().item()
        img_w = torch.ceil(w_max - w_min).int().item()
        sizes = (img_h, img_w)

        tgt_grid, tgt_cell, tgt_mask, \
        ref_grid, ref_cell, ref_mask, \
        stit_grid_s, queries = get_ingredient(ref, mask_stit, H_tgt2ref, eye, sizes)

        preds = model(
            ref, ref_grid, ref_cell, ref_mask,
            tgt, tgt_grid, tgt_cell, tgt_mask,
            stit_grid_s, sizes
        )

        gt = gt_stit.flatten(2).permute(0,2,1)[:, queries]
        loss = loss_fn(gt, preds)

        if torch.isnan(loss):
            gc.collect()
            torch.cuda.empty_cache()

        tot_loss += loss.item()
        pbar.set_description_str(desc="[Train] Loss: {:.4f}".format(loss.item()), refresh=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None

    return tot_loss / (b_id + 1)


def valid(valid_loader, model):
    model.eval()
    tot_psnr = 0

    pbar = tqdm(range(len(valid_loader)), smoothing=0.9)
    valid_loader = iter(valid_loader)

    for b_id in pbar:
        batch = next(valid_loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        img = (batch['inp'] - 0.5) * 2
        b, _, h, w = img.shape

        ref, tgt, H_tgt2ref, eye, \
        gt_stit, mask_stit, sizes = utils.random_warp(img, box_size=48, ovl_rate=0.75)

        max_wh, min_wh = sizes
        (w_max, h_max), (w_min, h_min) = max_wh, min_wh

        img_h = torch.ceil(h_max - h_min).int().item()
        img_w = torch.ceil(w_max - w_min).int().item()
        sizes = (img_h, img_w)

        tgt_grid, tgt_cell, tgt_mask, \
        ref_grid, ref_cell, ref_mask, \
        stit_grid_s, queries = get_ingredient(ref, mask_stit, H_tgt2ref, eye, sizes)

        pred = model(
            ref, ref_grid, ref_cell, ref_mask,
            tgt, tgt_grid, tgt_cell, tgt_mask,
            stit_grid_s, sizes
        ).cpu().numpy()

        gt = gt_stit.cuda().permute(0,2,3,1).reshape(b,-1,3)
        gt = gt[:, queries].cpu().numpy()
        tot_psnr += compare_psnr(gt, pred, data_range=2.)

        pbar.set_description_str(desc="PSNR:{:.4f}".format(tot_psnr/(b_id+1)), refresh=True)

        pred = None; loss = None

    return tot_psnr/(b_id+1)


def main(config_, save_path, args):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, valid_loader = make_data_loaders()

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training(resume=args.resume)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    best = 1e-8
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, epoch)
        with torch.no_grad():
            valid_psnr = valid(valid_loader, model)

        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('loss: {:.4f}, PSNR: {:.4f}'.format(train_loss, valid_psnr))

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
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

        if valid_psnr > best:
            best = valid_psnr
            torch.save(sv_file, os.path.join(save_path, 'best.pth'.format(epoch)))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args)
