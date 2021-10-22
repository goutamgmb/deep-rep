# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F
from actors.base_actor import BaseActor
from models.loss.spatial_color_alignment import SpatialColorAlignment
import models.layers.warp as lispr_warp


class DeepRepSRSyntheticActor(BaseActor):
    """Actor for training DeepRep model on synthetic burst dataset """
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        # Run network
        pred, aux_dict = self.net(data['burst'])

        # Compute loss
        loss_rgb_raw = self.objective['rgb'](pred, data['frame_gt'])
        loss_rgb = self.loss_weight['rgb'] * loss_rgb_raw

        if 'psnr' in self.objective.keys():
            psnr = self.objective['psnr'](pred.clone().detach(), data['frame_gt'])

        loss = loss_rgb

        stats = {'Loss/total': loss.item(),
                 'Loss/rgb': loss_rgb.item(),
                 'Loss/raw/rgb': loss_rgb_raw.item()}

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return loss, stats


class DeepRepSRBurstSRActor(BaseActor):
    """Actor for training DeepRep model on real-world BurstSR dataset using an aligned loss """
    def __init__(self, net, objective, alignment_net, loss_weight=None, sr_factor=4):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}

        self.sca = SpatialColorAlignment(alignment_net.eval(), sr_factor=sr_factor)
        self.loss_weight = loss_weight

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)
        self.sca.to(device)

        if 'lpips' in self.objective.keys():
            self.objective['lpips'].to(device)

    def __call__(self, data):
        # Run network
        gt = data['frame_gt']
        burst = data['burst']

        pred, aux_dict = self.net(burst)

        # Perform spatial and color alignment
        pred_warped_m, valid = self.sca(pred, gt, burst)

        # Compute loss
        loss_rgb_raw = self.objective['rgb'](pred_warped_m, gt, valid=valid)

        loss_rgb = self.loss_weight['rgb'] * loss_rgb_raw

        if 'psnr' in self.objective.keys():
            # detach, otherwise there is memory leak
            psnr = self.objective['psnr'](pred_warped_m.clone().detach(), gt, valid=valid)

        if 'mssim' in self.objective.keys():
            mssim = self.objective['mssim'](pred_warped_m.clone().detach(), gt)

        if 'lpips' in self.objective.keys():
            lpips = self.objective['lpips'](pred_warped_m.clone().detach(), gt)

        loss = loss_rgb

        stats = {'Loss/total': loss.item(),
                 'Loss/rgb': loss_rgb.item(),
                 'Loss/raw/rgb': loss_rgb_raw.item()}

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        if 'mssim' in self.objective.keys():
            stats['Stat/mssim'] = mssim.item()

        if 'lpips' in self.objective.keys():
            stats['Stat/lpips'] = lpips.item()
        return loss, stats


class DeepRepDenoisingActor(BaseActor):
    """Actor for training DeepRep model on burst denoising datasets """
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'rgb': 1.0}
        self.loss_weight = loss_weight

    def _convert_dict(self, meta_info_dict, batch_sz):
        meta_info_conv = []
        for b_elem in range(batch_sz):
            b_info = {}
            for k, v in meta_info_dict.items():
                b_info[k] = v[b_elem]
                if isinstance(b_info[k], torch.Tensor):
                    b_info[k].requires_grad = False
            meta_info_conv.append(b_info)
        return meta_info_conv

    def _get_offsets_gt(self, data):
        # Find flow, in (x, y) form
        # Note we are going from base image to shifted image
        offsets_gt = data['shifts'][:, 1:, ...] - data['shifts'][:, :1, ...]
        offsets_gt = offsets_gt.view(-1, *offsets_gt.shape[-3:])

        return offsets_gt

    def __call__(self, data):
        # Run network
        batch_sz = data['burst'].shape[0]

        if 'shifts' in data:
            offsets_gt = self._get_offsets_gt(data)
        else:
            offsets_gt = None

        noise_estimate = data.get('sigma_estimate', None)
        pred, aux_dict = self.net(data['burst'], noise_estimate=noise_estimate)

        # Compute loss
        loss_rgb_raw = self.objective['rgb'](pred, data['frame_gt'])
        loss_rgb = self.loss_weight['rgb'] * loss_rgb_raw

        loss_offsets = 0.0
        if 'offsets' in aux_dict and 'offsets' in self.loss_weight:
            offsets_pred = aux_dict['offsets']
            offsets_pred = offsets_pred.view(-1, *offsets_pred.shape[-3:])

            if offsets_pred.shape[-1] != offsets_gt.shape[-1]:
                offsets_pred = F.interpolate(offsets_pred, size=(offsets_gt.shape[-1], offsets_gt.shape[-2]),
                                             mode='bilinear', align_corners=False) * offsets_gt.shape[-1] / offsets_pred.shape[-1]
            loss_offsets_raw = self.objective['offsets'](offsets_pred, offsets_gt)
            loss_offsets = self.loss_weight['offsets'] * loss_offsets_raw

        loss_photometric = 0.0
        if 'offsets' in aux_dict and 'photometric' in self.loss_weight:
            # Unsupervised photometric loss for training the optical flow network.
            # Warp the reference frame to other frames using the estimated flow, and minimize the error between
            # warped image and the original images
            offsets_pred = aux_dict['offsets']
            offsets_pred = offsets_pred.view(-1, *offsets_pred.shape[-3:])

            if offsets_pred.shape[-1] != offsets_gt.shape[-1]:
                offsets_pred = F.interpolate(offsets_pred, size=(offsets_gt.shape[-1], offsets_gt.shape[-2]),
                                             mode='bilinear', align_corners=False) * offsets_gt.shape[-1] / \
                               offsets_pred.shape[-1]

            base_frame = data['burst'][:, :1].contiguous()
            oth_frames = data['burst'][:, 1:].contiguous()
            base_frame = base_frame.repeat(1, oth_frames.shape[1], 1, 1, 1)

            base_frame = base_frame.view(-1, *base_frame.shape[-3:])
            oth_frames = oth_frames.view(-1, *oth_frames.shape[-3:])
            base_frame_warped = lispr_warp.warp(base_frame, offsets_pred)

            loss_photometric_raw = self.objective['photometric'](base_frame_warped, oth_frames)
            loss_photometric = self.loss_weight['photometric'] * loss_photometric_raw

        if 'psnr' in self.objective.keys():
            # detach, otherwise there is memory leak
            psnr = self.objective['psnr'](pred.clone().detach(), data['frame_gt'])

        loss = loss_rgb + loss_offsets + loss_photometric

        stats = {'Loss/total': loss.item(),
                 'Loss/rgb': loss_rgb.item(),
                 'Loss/raw/rgb': loss_rgb_raw.item()}

        if 'offsets' in aux_dict and 'offsets' in self.loss_weight:
            stats['Loss/offset'] = loss_offsets.item()
            stats['Loss/raw/offset'] = loss_offsets_raw.item()

        if 'offsets' in aux_dict and 'photometric' in self.loss_weight:
            stats['Loss/photometric'] = loss_photometric.item()
            stats['Loss/raw/photometric'] = loss_photometric_raw.item()

        if 'psnr' in self.objective.keys():
            stats['Stat/psnr'] = psnr.item()

        return loss, stats
