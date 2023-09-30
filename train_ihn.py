import os
import gc
import yaml
import utils
import torch
import models
import argparse
import datasets
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.manual_seed(0)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    valid_loader = make_data_loader(config.get('valid_dataset'), tag='valid')
    return train_loader, valid_loader


def prepare_train(resume=False, finetune=False):
    if resume or finetune:
        H_model = models.IHN().cuda()
        sv_file = torch.load(config['resume_align'])
        H_model.load_state_dict(sv_file['model']['sd'])

        if not finetune:
            epoch_start = sv_file['epoch'] + 1
            optimizer = utils.make_optimizer(H_model.parameters(), sv_file['optimizer'], load_sd=True)

        else:
            epoch_start = 1
            optimizer = utils.make_optimizer(H_model.parameters(), config['optimizer'])

        if config.get('multi_step_lr') is None:
            lr_scheduler = None

        else:
            lr_scheduler = ExponentialLR(optimizer, gamma=0.98)

    else:
        H_model = models.IHN().cuda()
        optimizer = utils.make_optimizer(H_model.parameters(), config['optimizer'])

        epoch_start = 1

        if config.get('multi_step_lr') is None:
            lr_scheduler = None

        else:
            lr_scheduler = ExponentialLR(optimizer, gamma=0.98)

    num_params = utils.compute_num_params(H_model)

    if num_params >= 1e6:
        num_params = '{:.1f}M'.format(num_params / 1e6)
    else:
        num_params = '{:.1f}K'.format(num_params / 1e3)

    log('model params={}'.format(str(num_params)))

    return H_model, optimizer, epoch_start, lr_scheduler


def train(loader, model, optimizer, epoch, finetune=False):
    model.train()

    unsup = True

    tot_loss = 0
    l1_loss = nn.L1Loss(reduction='mean')

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)
    print('[Scheduler] lr: {}'.format(optimizer.param_groups[0]['lr']))

    for b_id in pbar:
        batch = next(loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        if not unsup and not finetune:
            inp_tgt = batch['inp_tgt']
            inp_src = batch['inp_src']
            gt_shift = batch['shift']

            b, c, h, w = inp_tgt.shape
            inp_tgt_ = inp_tgt.contiguous() * 255
            inp_src_ = inp_src.contiguous() * 255

            four_pred, disps = model(inp_src_, inp_tgt_, iters_lev0=6, iters_lev1=3)

            loss = 0
            for i in range(len(disps)):
                loss += l1_loss(gt_shift, disps[-i-1]) * 0.85 ** i

            pbar.set_description_str(
                desc="[Train] Epoch:{}/{}, Loss: {:.4f}".format(
                    epoch, config['epoch_max'], loss.item()), refresh=True)

        else:
            inp_tgt = batch['inp_tgt']
            inp_src = batch['inp_src']

            mask = torch.ones_like(inp_src).cuda()

            b, c, h, w = inp_tgt.shape
            inp_tgt_ = inp_tgt.contiguous() * 255
            inp_src_ = inp_src.contiguous() * 255

            four_pred, disps = model(inp_tgt_, inp_src_, iters_lev0=6, iters_lev1=3)

            loss = 0
            for i in range(len(disps)):
                shift = disps[i].view(b, 2, -1).permute(0, 2, 1)

                H, _, _, _, _ = utils.get_H(shift, (h, w))
                H = utils.compens_H(H, size=(h, w))

                tgt_w_ovl = utils.STN(inp_tgt, torch.inverse(H))
                mask_tgt_w = utils.STN(mask, torch.inverse(H)).round()
                src_ovl = inp_src * mask_tgt_w

                loss += l1_loss(tgt_w_ovl, src_ovl) * 0.85 ** (len(disps) - i - 1)

            pbar.set_description_str(
                desc="[Train] Epoch:{}/{}, Loss: {:.4f}".format(
                    epoch, config['epoch_max'], loss.item()), refresh=True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, norm_type=2)
        optimizer.step()

        pred, loss = None, None

    return tot_loss / (b_id + 1)


def valid(loader, model, finetune=False):
    model.eval()

    tot_mace = 0
    tot_psnr = 0

    l1_loss = nn.L1Loss(reduction='mean')

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)

    for b_id in pbar:
        batch = next(loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        if not finetune:
            inp_tgt = batch['inp_tgt']
            inp_src = batch['inp_src']
            gt_shift = batch['shift']

            inp_tgt_ = inp_tgt.contiguous() * 255
            inp_src_ = inp_src.contiguous() * 255
            shift, _ = model(inp_src_, inp_tgt_, iters_lev0=6, iters_lev1=3)

            mace = (gt_shift - shift)**2
            mace = torch.mean((mace[:,:,0] + mace[:,:,1])**0.5)
            tot_mace += mace.item()

            pbar.set_description_str(desc="[Valid] MACE:{:.4f}".format(tot_mace/(b_id + 1)), refresh=True)

        else:
            inp_tgt = batch['inp_tgt']
            inp_src = batch['inp_src']

            b, c, h, w = inp_tgt.shape
            inp_tgt_ = inp_tgt.contiguous() * 255
            inp_src_ = inp_src.contiguous() * 255
            mask = torch.ones_like(inp_src).cuda()

            four_pred, _ = model(inp_src_, inp_tgt_, iters_lev0=6, iters_lev1=3)
            shift = four_pred.view(b, 2, -1).permute(0, 2, 1)

            shape=(128, 128)
            H, _, _, _, _ = utils.get_H(shift, shape)
            H = utils.compens_H(H, size=shape)

            tgt_w_ovl = utils.STN(inp_tgt, torch.inverse(H))
            mask_tgt_w = utils.STN(mask, torch.inverse(H)).round()
            src_ovl = inp_src * mask_tgt_w

            psnr = compare_psnr(tgt_w_ovl.cpu().numpy(), src_ovl.cpu().numpy(), data_range=1.)
            tot_psnr += psnr

            pbar.set_description_str(desc="[Valid] PSNR:{:.4f}".format(tot_psnr/(b_id + 1)), refresh=True)

    return tot_mace / (b_id + 1), tot_psnr / (b_id + 1)


def main(config_, save_path, args):
    global config, log
    config = config_

    log, _ = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, valid_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_train(resume=args.resume, finetune=args.finetune)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')
    timer = utils.Timer()

    best_psnr = 1e-4
    best_mace = 1e+4

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        loss = train(train_loader, model, optimizer, epoch, finetune=args.finetune)
        with torch.no_grad():
            val_mace, val_psnr = valid(valid_loader, model, finetune=args.finetune)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if n_gpus > 1:
            model_ = model.module

        else:
            model_ = model

        model_spec = config['H_model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        torch.save(sv_file, os.path.join(save_path, 'H-epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, 'H-epoch-{}.pth'.format(epoch)))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_psnr_e = epoch
            torch.save(sv_file, os.path.join(save_path, 'H-finetune-best.pth'))

        if val_mace < best_mace:
            best_mace = val_mace
            best_mace_e = epoch
            torch.save(sv_file, os.path.join(save_path, 'H-best.pth'))

        if not args.finetune:
            print('[{}/{} Summary] Align:{:.2f}, MACE:{:.3f}'.format(epoch, epoch_max, loss, val_mace))
            record = '[Record] Best MACE : {} Epoch - {:.3f}'.format(best_mace_e, best_mace)

        else:
            print('[{}/{} Summary] Align:{:.2f}, PSNR:{:.3f}'.format(epoch, epoch_max, loss, val_psnr))
            record = '[Record] Best PSNR : {} Epoch - {:.3f}'.format(best_psnr_e, best_psnr)

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

        record += '\t{} {}/{}'.format(t_epoch, t_elapsed, t_all)
        log(record)

        print('=' * os.get_terminal_size().columns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    save_path = os.path.join('./save', save_name)
    main(config, save_path, args)
