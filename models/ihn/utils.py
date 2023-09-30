import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys

from skimage import io
from scipy import interpolate
from kornia.geometry.transform import get_perspective_transform as get_H_mat

def compute_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)

def save_img(img, path):
    npimg = img.detach().cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.astype(np.uint8)
    io.imsave(path, npimg)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask

class Logger_(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_mace_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])
        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        print(training_str + metrics_str + time_left_hms)
        # logging running loss to total loss
        self.train_mace_list.append(np.mean(self.running_loss_dict['mace']))
        self.train_steps_list.append(self.total_steps)
        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []
            self.running_loss_dict[key].append(metrics[key])
        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}


def compens_H(H, size):
    h, w = size
    b, _, _ = H.shape

    M = torch.Tensor([[(h-1)/2., 0.,   (h-1)/2.],
                      [0.,   (w-1)/2., (w-1)/2.],
                      [0.,   0.,   1.]]).unsqueeze(0).cuda()

    M = M.repeat(b, 1, 1)
    M_inv = torch.inverse(M)
    return M_inv @ H @ M


def get_H(shift, size):
    h, w  = size
    b, _, _ = shift.shape

    src_corner = torch.Tensor([[[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]])
    src_corner = src_corner.repeat(b, 1, 1)
    tgt_corner = src_corner + shift.cpu()

    H_src2tgt = get_H_mat(src_corner, tgt_corner).cuda()
#    H_src2tgt = get_H_mat(tgt_corner, src_corner).cuda()

    return H_src2tgt


def STN(image2_tensor, H_tf, offsets=()):
    """Spatial Transformer Layer"""

    def _repeat(x, n_repeats):
        rep = torch.ones(1, n_repeats, dtype=x.dtype)
        x = torch.mm(x.reshape(-1, 1), rep)
        return x.reshape(-1)

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, channels, height, width = im.shape
        device = im.device

        x, y = x.float().to(device), y.float().to(device)
        height_f, width_f = torch.tensor(height).float(), torch.tensor(width).float()
        out_height, out_width = out_size

        # scale indices from [-1, 1] to [0, width/height]
        # effect values will exceed [-1, 1], so clamp is unnecessary or even incorrect
        x = (x + 1.0) * (width_f-1) / 2.0
        y = (y + 1.0) * (height_f-1) / 2.0

        # do sampling
        x0 = x.floor().int()
        x1 = x0 + 1
        y0 = y.floor().int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)

        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(num_batch) * dim1, out_height * out_width).to(device)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.permute(0, 2, 3, 1).reshape(-1, channels).float()
        Ia, Ib, Ic, Id = im_flat[idx_a], im_flat[idx_b], im_flat[idx_c], im_flat[idx_d]

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output  # .clamp(0., 1.) stupid

    def _meshgrid(height, width):
        x_t = torch.mm(torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).unsqueeze(0))
        y_t = torch.mm(torch.linspace(-1.0, 1.0, height).unsqueeze(1), torch.ones(1, width))

        x_t_flat = x_t.reshape(1, -1)
        y_t_flat = y_t.reshape(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
        return grid

    bs, nc, height, width = image2_tensor.shape
    device = image2_tensor.device

    is_nan = torch.isnan(H_tf.reshape(bs, 9)).any(dim=1)
    assert is_nan.sum() == 0, f'{image2_tensor.shape} {len(offsets)}, {[off.view(-1, 8)[is_nan] for off in offsets]}'
    H_tf = H_tf.reshape(-1, 3, 3).float()
    # grid of (x_t, y_t, 1)
    grid = _meshgrid(height, width).unsqueeze(0).expand(bs, -1, -1).to(device)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = torch.bmm(H_tf, grid)  # [bs,3,3] x [bs,3,w*h] -> [bs,3,w*h]
    x_s, y_s, t_s = torch.chunk(T_g, 3, dim=1)
    # The problem may be here as a general homo does not preserve the parallelism
    # while an affine transformation preserves it.
    t_s_flat = t_s.reshape(-1)
    eps, maximal = 1e-2, 10.
    t_s_flat[t_s_flat.abs() < eps] = eps
    # 1.25000 / 1.38283e-05 = inf   in float16 (6.55e4)

    #  batchsize * width * height
    x_s_flat = x_s.reshape(-1) / t_s_flat
    y_s_flat = y_s.reshape(-1) / t_s_flat

    input_transformed = _interpolate(image2_tensor, x_s_flat, y_s_flat, (height, width))

    output = input_transformed.reshape(bs, height, width, nc).permute(0, 3, 1, 2)

    return output
