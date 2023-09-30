import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class SeamEvaluator(nn.Module):
    def __init__(self):
        super(SeamEvaluator, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

        self.seam_filters = torch.Tensor(
            [[1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0]]
        ).reshape(1,1,3,3).cuda()

        sobel_x = [[-1.,  0., 1.],
                   [-2.,  0., 2.],
                   [-1.,  0., 1.]]

        sobel_y = [[-1., -2., -1.],
                   [ 0.,  0.,  0.],
                   [ 1.,  2.,  1.]]

        self.sobel_x = torch.FloatTensor(sobel_x).reshape(1, 1, 3, 3).cuda()
        self.sobel_y = torch.FloatTensor(sobel_y).reshape(1, 1, 3, 3).cuda()


    def edge_extraction(self, gen_frames):
        b, c, _, _ = gen_frames.shape
        gen_dx = F.conv2d(gen_frames, self.sobel_x, bias=None, stride=1, padding=1).abs()
        gen_dy = F.conv2d(gen_frames, self.sobel_y, bias=None, stride=1, padding=1).abs()
        edge = (gen_dx + gen_dy).clamp_(0, 1)
        return edge


    def get_seam(self, img):
        seam = self.edge_extraction(torch.mean(img, dim=1, keepdim=True))
        fea1 = F.conv2d(seam, self.seam_filters, bias=None, stride=1, padding=1).clamp_(0, 1)
        fea2 = F.conv2d(fea1, self.seam_filters, bias=None, stride=1, padding=1).clamp_(0, 1)
        fea3 = F.conv2d(fea2, self.seam_filters, bias=None, stride=1, padding=1).clamp_(0, 1)
        return fea3


    def get_seam_mask(self, mask1, mask2):
        seam_m = mask1 * self.get_seam(mask2)
        return seam_m
