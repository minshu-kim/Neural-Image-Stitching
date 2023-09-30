import os
import cv2
import time
import math
import shutil

import torch
import random
import numpy as np
import torch.nn.functional as F

from srwarp import warp
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, AdamW

from srwarp import transform as transform_w
from kornia.geometry.transform import get_perspective_transform as get_H_mat


class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None
def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def random_warp(ref, box_size=128, ovl_rate=0.5, offset_ratio=0.20):
    b, c, h, w = ref.shape

    box = box_size/2
    rho = int(box_size * offset_ratio)

    lb = int(box_size * (ovl_rate - 1))
    ub = int(box_size * (1 - ovl_rate))

    corners = torch.Tensor([[w/2-box, h/2-box], [w/2+box, h/2-box], [w/2-box, h/2+box], [w/2+box, h/2+box]])
    tgt_offset = torch.randint(low=lb, high=ub, size=(1,2))

    rand_corners = corners + tgt_offset
    shifts = torch.randint(low=-rho, high=rho, size=(4,2))

    rand_corners_w = rand_corners + shifts
    H_ref2tgt = get_H_mat(rand_corners_w.view(1,4,2), rand_corners.view(1,4,2)).double()

    tgt = warp.warp_by_function(
        ref,
        transform_w.inverse_3x3(H_ref2tgt[0]),
        sizes=(h,w),
        kernel_type='bicubic',
        adaptive_grid=True,
        fill=-255,
    )

    tgt_corners = rand_corners - rand_corners[0]
    shifts = rand_corners_w - corners
    ref_corners = tgt_corners + shifts
    all_corners = torch.cat([ref_corners, tgt_corners], dim=0).permute(1, 0)

    H_tgt2ref = get_H_mat(tgt_corners.view(1,4,2), ref_corners.view(1,4,2)).double()

    max_wh = torch.max(all_corners, dim=-1)[0]
    min_wh = torch.min(all_corners, dim=-1)[0]

    sizes = (max_wh, min_wh)
    max_w, max_h = max_wh.int(); min_w, min_h = min_wh.int()

    patch_x = corners[0][0].int()
    patch_y = corners[0][1].int()
    patch_x_w = rand_corners[0][0].int()
    patch_y_w = rand_corners[0][1].int()

    bias_wh = torch.min(corners.permute(1,0), dim=-1)[0]
    bias_w, bias_h = bias_wh.int()

    T = get_translation(min_h, min_w).double()
    eye = (T @ torch.eye(3).double()).double()
    H_tgt2ref = (T @ H_tgt2ref[0]).double()

    img_h, img_w = max_h-min_h, max_w-min_w
    ones = torch.ones(b, c, box_size, box_size).cuda()

    ref_ = ref[:, :, patch_y : patch_y+box_size, patch_x : patch_x+box_size]
    tgt_ = tgt[:, :, patch_y_w : patch_y_w+box_size, patch_x_w : patch_x_w+box_size]
    gt_stit = ref[:,:, bias_h+min_h : bias_h+max_h, bias_w+min_w : bias_w+max_w]

    tgt_mask = warp.warp_by_function(
        ones,
        transform_w.inverse_3x3(H_tgt2ref),
        sizes=(max_h-min_h, max_w-min_w),
        kernel_type='bicubic',
        adaptive_grid=False,
        fill=0,
    )

    ref_mask = warp.warp_by_function(
        ones,
        transform_w.inverse_3x3(eye),
        sizes=(max_h-min_h, max_w-min_w),
        kernel_type='bicubic',
        adaptive_grid=False,
        fill=0,
    )

    stit_mask = (tgt_mask + ref_mask).clamp(0, 1)
    gt_stit *= stit_mask

#    gt_ref = gt_stit.contiguous() * ref_mask
#    gt_tgt = gt_stit.contiguous() * tgt_mask

    sizes = (max_wh, min_wh)

    return ref_, tgt_, H_tgt2ref, eye, gt_stit, stit_mask, sizes


def get_translation(h_min, w_min):
    trans = torch.Tensor([[1., 0., -w_min],
                          [0., 1., -h_min],
                          [0., 0., 1.]]).double()

    return trans


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_cell(coord, img, sizes=None):
    if sizes is not None:
        h, w = sizes
        coord_bot_left  = coord + torch.tensor([-1/h, -1/w]).unsqueeze(0)
        coord_bot_right = coord + torch.tensor([-1/h,  1/w]).unsqueeze(0)
        coord_top_left  = coord + torch.tensor([ 1/h, -1/w]).unsqueeze(0)
        coord_top_right = coord + torch.tensor([ 1/h,  1/w]).unsqueeze(0)
        coord_left  = coord + torch.tensor([-1/h, 0]).unsqueeze(0)
        coord_right = coord + torch.tensor([ 1/h, 0]).unsqueeze(0)
        coord_bot   = coord + torch.tensor([ 0, -1/w]).unsqueeze(0)
        coord_top   = coord + torch.tensor([ 0,  1/w]).unsqueeze(0)

    else:
        coord_bot_left  = coord + torch.tensor([-1/img.shape[-2], -1/img.shape[-1]]).unsqueeze(0)
        coord_bot_right = coord + torch.tensor([-1/img.shape[-2],  1/img.shape[-1]]).unsqueeze(0)
        coord_top_left  = coord + torch.tensor([ 1/img.shape[-2], -1/img.shape[-1]]).unsqueeze(0)
        coord_top_right = coord + torch.tensor([ 1/img.shape[-2],  1/img.shape[-1]]).unsqueeze(0)
        coord_left  = coord + torch.tensor([-1/img.shape[-2], 0]).unsqueeze(0)
        coord_right = coord + torch.tensor([ 1/img.shape[-2], 0]).unsqueeze(0)
        coord_bot   = coord + torch.tensor([ 0, -1/img.shape[-1]]).unsqueeze(0)
        coord_top   = coord + torch.tensor([ 0,  1/img.shape[-1]]).unsqueeze(0)

    cell_side   = torch.cat((coord_left, coord_right, coord_bot, coord_top), dim=0)
    cell_corner = torch.cat((coord_bot_left, coord_bot_right, coord_top_left, coord_top_right), dim=0)
    cell = torch.cat((cell_corner, cell_side, coord), dim=0)
    return cell


def to_pixel_samples(img, sizes=None):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    if sizes is not None:
        coord = make_coord(sizes)
    else:
        coord = make_coord(img.shape[-2:])

    if img is not None:
        rgb = img.view(3, -1).permute(1, 0)
        return coord, rgb

    return coord


def gridy2gridx_homography(gridy, H, W, h, w, m, cpu=True):
    # scaling
    gridy += 1
    gridy[:, 0] *= H / 2
    gridy[:, 1] *= W / 2
    gridy -= 0.5
    gridy = gridy.flip(-1)

    # coord -> homogeneous coord
    if cpu:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1)), dim=-1).double()
    else:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1).cuda()), dim=-1).double()
    # transform
    if cpu:
        m = transform_w.inverse_3x3(m)
    else:
        m = transform_w.inverse_3x3(m).cuda()

    gridx = torch.mm(m, gridy.permute(1, 0)).permute(1, 0)

    # homogeneous coord -> coord
    gridx[:, 0] /= gridx[:, -1]
    gridx[:, 1] /= gridx[:, -1]
    gridx = gridx[:, 0:2]


    # rescaling
    gridx = gridx.flip(-1)
    gridx += 0.5
    gridx[:, 0] /= h / 2
    gridx[:, 1] /= w / 2
    gridx -= 1
    gridx = gridx.float()

    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask = mask.float()

    return gridx, mask


def celly2cellx_homography(celly, H, W, h, w, m, cpu=True, rescale_m=False):
    cellx, _ = gridy2gridx_homography(celly, H, W, h, w, m, cpu) # backward mapping
    return shape_estimation(cellx)


def shape_estimation(cell):
    cell_1 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :] \
                - cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]
    cell_2 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :] \
                - cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :] # Jacobian
    cell_3 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :] \
              - 2*cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] \
                + cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]
    cell_4 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :] \
              - 2*cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] \
                + cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :] # Second-order derivatives in Hessian
    cell_5 = cell[3*cell.shape[0]//9:4*cell.shape[0]//9, :] \
                - cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] \
                - cell[1*cell.shape[0]//9:2*cell.shape[0]//9, :] \
                + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] \
                - cell[2*cell.shape[0]//9:3*cell.shape[0]//9, :] \
                + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] # Cross-term in Hessian
    shape = torch.cat((cell_1, cell_2, 4*cell_3, 4*cell_4, cell_5), dim=-1)
    return shape


def quantize(x: torch.Tensor) -> torch.Tensor:
    x = 127.5 * (x + 1)
    x = x.clamp(min=0, max=255)
    x = x.round()
    x = x / 127.5 - 1
    return x


def compens_H(H, size):
    h, w = size

    M = torch.Tensor([[(h-1)/2., 0.,   (h-1)/2.],
                      [0.,   (w-1)/2., (w-1)/2.],
                      [0.,   0.,   1.]]).cuda()

    M_inv = torch.inverse(M)
    return M_inv @ H @ M


def get_H(shift, size):
    h, w  = size
    b, _, _ = shift.shape

    tgt_corner = torch.Tensor([[[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]])
    tgt_corner = tgt_corner.repeat(b, 1, 1)
    ref_corner = tgt_corner + shift.cpu()
    H_tgt2ref = get_H_mat(tgt_corner, ref_corner).cuda()

    corners = torch.cat([tgt_corner.view(b, -1), ref_corner.view(b, -1)])
    corners_x = corners[:, ::2]
    corners_y = corners[:, 1::2]

    width_max = torch.max(corners_x)
    height_max = torch.max(corners_y)
    width_min = torch.min(corners_x)
    height_min = torch.min(corners_y)

    return H_tgt2ref, width_max, width_min, height_max, height_min


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

    x_s_flat = x_s.reshape(-1) / t_s_flat
    y_s_flat = y_s.reshape(-1) / t_s_flat

    input_transformed = _interpolate(image2_tensor, x_s_flat, y_s_flat, (height, width))
    output = input_transformed.reshape(bs, height, width, nc).permute(0, 3, 1, 2)

    return output
