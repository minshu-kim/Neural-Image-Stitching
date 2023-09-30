import gc
import torch
import models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models import register
from utils import make_coord


@register('NIS')
class NeuralImageStitching(nn.Module):
    def __init__(self, encoder_spec, blender_spec, hidden_dim=256, imnet_spec=None):
        super().__init__()

        self.encoder = models.make(encoder_spec)
        self.blender = models.make(blender_spec)

        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Conv2d(10, hidden_dim//2, 1, bias=False)

        self.imnet = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 3),
        )


    def gen_feat(self, inp):
        feat_coord = make_coord(inp.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])

        feat = self.encoder(inp)
        coeff = self.coef(feat)
        freqq = self.freq(feat)

        return feat, feat_coord, coeff, freqq


    def gen_feat_for_blender(self, inp):
        feat = self.blender(inp)

        return feat


    def NeuralWarping(self, img, feat, feat_coord, freq, coef, coord, cell, sizes):
        h, w = sizes
        b = img.shape[0]
        coord = coord.reshape(b, h, w, 2)

        w_coef = F.grid_sample(
            coef,
            coord.flip(-1),
            mode='nearest',
            align_corners=False
        )

        w_freq = F.grid_sample(
            freq,
            coord.flip(-1),
            mode='nearest',
            align_corners=False
        ).permute(0, 2, 3, 1)

        w_coord = F.grid_sample(
            feat_coord,
            coord.flip(-1),
            mode='nearest',
            align_corners=False
        ).permute(0, 2, 3, 1)

        rel_coord = coord - w_coord
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]

        rel_cell = cell.clone()
        rel_cell[..., [0, 2, 4, 6, 8]] *= feat.shape[-2]
        rel_cell[..., [1, 3, 5, 7, 9]] *= feat.shape[-1]
        rel_cell = rel_cell.reshape(b, *sizes, 10).permute(0, 3, 1, 2)

        w_freq = torch.stack(torch.split(w_freq, 2, dim=-1), dim=-1)
        w_freq = torch.mul(w_freq, rel_coord.unsqueeze(-1))
        w_freq = torch.sum(w_freq, dim=-2).permute(0, 3, 1, 2)
        w_freq += self.phase(rel_cell)
        w_freq = torch.cat((torch.cos(np.pi*w_freq), torch.sin(np.pi*w_freq)), dim=1)

        return  torch.mul(w_coef, w_freq)


    def query_rgb(self, inp, coord):
        q_inp = F.grid_sample(
            inp, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        bs, q, _ = q_inp.shape
        pred = self.imnet(q_inp.contiguous().view(bs * q, -1)).view(bs, q, -1)

        return pred


    def forward(self, ref, ref_grid, ref_cell, ref_mask,
                tgt, tgt_grid, tgt_cell, tgt_mask,
                stit_coord_s, sizes, eval_mode=False):

        fea_tgt, fea_tgt_grid, tgt_coef, tgt_freq = self.gen_feat(tgt)
        fea_tgt_w = self.NeuralWarping(
            tgt, fea_tgt, fea_tgt_grid,
            tgt_freq, tgt_coef, tgt_grid, tgt_cell, sizes
        )

        fea_ref, fea_ref_grid, ref_coef, ref_freq = self.gen_feat(ref)
        fea_ref_w = self.NeuralWarping(
            ref, fea_ref, fea_ref_grid,
            ref_freq, ref_coef, ref_grid, ref_cell, sizes
        )

        if eval_mode:
            black = torch.zeros_like(fea_tgt_w).cuda()
            fea = torch.cat([black, fea_tgt_w * tgt_mask], dim=1)

        else:
            fea = torch.cat([fea_ref_w * ref_mask, fea_tgt_w * tgt_mask], dim=1)

        stit_rep = self.gen_feat_for_blender(fea)
        im = self.query_rgb(stit_rep, stit_coord_s)

        return im
