import cv2
import math
import torch
import random
import functools
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset

from datasets import register
from utils import to_pixel_samples, make_coord, make_cell, gridy2gridx_homography, celly2cellx_homography


@register('base')
class Paired(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        return {
            'img': img,
        }


@register('paired-images')
class Paired(Dataset):
    def __init__(self, dataset, scaling=False, max_scale=1/0.35, scale=None):
        self.dataset = dataset
        self.scaling = scaling
        self.max_scale = max_scale
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_ref, img_tgt = self.dataset[idx]

        return {
            'inp_ref'  : img_ref,
            'inp_tgt'  : img_tgt,
        }


marginal = 32
patch_size = 128

@register('ihn-onthefly')
class WarpYspaceCoordCell(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        img1 = np.copy(img)
        img2 = np.copy(img1)

        (height, width, _) = img1.shape

        y = random.randint(marginal, height - marginal - patch_size)
        x = random.randint(marginal, width - marginal - patch_size)

        top_left_point = (x, y)
        bottom_right_point = (patch_size + x, patch_size + y)

        perturbed_four_points_cord = []

        top_left_point_cord = (x, y)
        bottom_left_point_cord = (x, patch_size + y)
        bottom_right_point_cord = (patch_size + x, patch_size + y)
        top_right_point_cord = (x + patch_size, y)
        four_points_cord = [top_left_point_cord, bottom_left_point_cord, bottom_right_point_cord, top_right_point_cord]

        try:
            perturbed_four_points_cord = []
            for i in range(4):
                t1 = random.randint(-marginal, marginal)
                t2 = random.randint(-marginal, marginal)

                perturbed_four_points_cord.append((four_points_cord[i][0] + t1,
                                                  four_points_cord[i][1] + t2))

            y_grid, x_grid = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
            point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

            org = np.float32(four_points_cord)
            dst = np.float32(perturbed_four_points_cord)
            H = cv2.getPerspectiveTransform(org, dst)
            H_inverse = np.linalg.inv(H)
        except:
            perturbed_four_points_cord = []
            for i in range(4):
                t1 = 32//(i+1)
                t2 = -32//(i+1)

                perturbed_four_points_cord.append((four_points_cord[i][0] + t1,
                                                  four_points_cord[i][1] + t2))

            y_grid, x_grid = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
            point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

            org = np.float32(four_points_cord)
            dst = np.float32(perturbed_four_points_cord)

            H = cv2.getPerspectiveTransform(org, dst)
            H_inverse = np.linalg.inv(H)

        warped_image = cv2.warpPerspective(img2, H_inverse, (img1.shape[1], img1.shape[0]))

        img_patch_ori = img1[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0], :]
        img_patch_pert = warped_image[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0],:]

        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H).squeeze()

        diff_branch1 = point_transformed_branch1 - np.array(point, dtype=np.float64)
        diff_x_branch1 = diff_branch1[:, 0]
        diff_y_branch1 = diff_branch1[:, 1]

        diff_x_branch1 = diff_x_branch1.reshape((img1.shape[0], img1.shape[1]))
        diff_y_branch1 = diff_y_branch1.reshape((img1.shape[0], img1.shape[1]))

        pf_patch_x_branch1 = diff_x_branch1[top_left_point[1]:bottom_right_point[1],
                             top_left_point[0]:bottom_right_point[0]]

        pf_patch_y_branch1 = diff_y_branch1[top_left_point[1]:bottom_right_point[1],
                             top_left_point[0]:bottom_right_point[0]]

        pf_patch = np.zeros((patch_size, patch_size, 2))
        pf_patch[:, :, 0] = pf_patch_x_branch1
        pf_patch[:, :, 1] = pf_patch_y_branch1

        img_patch_ori = img_patch_ori[:, :, ::-1].copy()
        img_patch_pert = img_patch_pert[:, :, ::-1].copy()
        img = torch.from_numpy((img)).float().permute(2, 0, 1)
        img1 = torch.from_numpy((img_patch_ori)).float().permute(2, 0, 1)
        img2 = torch.from_numpy((img_patch_pert)).float().permute(2, 0, 1)
        flow = torch.from_numpy(pf_patch).permute(2, 0, 1).float()

        ### homo
        four_point_org = torch.zeros((2, 2, 2))
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([128, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, 128])
        four_point_org[:, 1, 1] = torch.Tensor([128, 128])

        four_point = torch.zeros((2, 2, 2))
        four_point[:, 0, 0] = flow[:, 0, 0] + torch.Tensor([0, 0])
        four_point[:, 0, 1] = flow[:, 0, -1] + torch.Tensor([128, 0])
        four_point[:, 1, 0] = flow[:, -1, 0] + torch.Tensor([0, 128])
        four_point[:, 1, 1] = flow[:, -1, -1] + torch.Tensor([128, 128])
        four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0)
        four_point = four_point.flatten(1).permute(1, 0).unsqueeze(0)

        gt_shift = torch.zeros(2,2,2)
        gt_shift[:, 0, 0] = flow[:, 0, 0]
        gt_shift[:, 0, 1] = flow[:, 0, -1]
        gt_shift[:, 1, 0] = flow[:, -1, 0]
        gt_shift[:, 1, 1] = flow[:, -1, -1]

        # prepare coordinate in Y space
        gridy, hr_rgb = to_pixel_samples(img1.contiguous())

        # prepare cell in Y space
        cell = make_cell(gridy, img1.contiguous())

        return {
            'inp': img,
            'inp_tgt': img1,
            'inp_src': img2,
            'shift': gt_shift,
            'cell': cell,
            'coord': gridy
        }


@register('warp-yspace-coord-cell')
class WarpYspaceCoordCell(Dataset):

    def __init__(self, dataset, inp_size=None, box_size=128, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.box_size = (box_size, box_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # crop image for batching
        img = self.dataset[idx]

        if self.inp_size is not None:
            x0 = random.randint(0, img.shape[-2] - self.inp_size)
            y0 = random.randint(0, img.shape[-1] - self.inp_size)
            img = img[:, x0: x0 + self.inp_size, y0: y0 + self.inp_size]

        # augmentation
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img = augment(img)

        # prepare coordinate in Y space
        gridy, hr_rgb = to_pixel_samples(img.contiguous())

        # prepare cell in Y space
        cell = make_cell(gridy, img)

        return {
            'inp': img,
            'cell': cell,
            'coord': gridy,
            'gt': hr_rgb
        }
