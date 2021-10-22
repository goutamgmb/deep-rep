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

import torch.nn as nn
import torch
import torch.nn.functional as F
import models.layers.blocks as blocks
import models.layers.warp as lispr_warp


class SimpleWeightPredictor(nn.Module):
    def __init__(self, input_dim, project_dim, offset_feat_dim,
                 num_offset_feat_extractor_res=1, num_weight_predictor_res=1,
                 use_bn=False, activation='relu', use_noise_estimate=False,
                 num_noise_feat_extractor_res=1, noise_feat_dim=32,
                 offset_modulo=None, use_offset=True,
                 ref_offset_noise=0.0, use_softmax=True, use_abs_diff=False, padding_mode='zeros', use_mean=False,
                 color_input=False):
        super().__init__()
        self.use_offset = use_offset
        self.offset_modulo = offset_modulo
        self.ref_offset_noise = ref_offset_noise
        self.use_softmax = use_softmax
        self.use_abs_diff = use_abs_diff
        self.use_mean = use_mean

        self.feat_project_layer = blocks.conv_block(input_dim, project_dim, 1, stride=1, padding=0,
                                                    batch_norm=use_bn,
                                                    activation=activation, padding_mode=padding_mode)
        self.use_noise_estimate = use_noise_estimate

        if self.use_noise_estimate:
            noise_feat_extractor = []
            if color_input:
                noise_est_dim = 3
            else:
                noise_est_dim = 1
            noise_feat_extractor.append(blocks.conv_block(noise_est_dim, noise_feat_dim, 3, stride=1, padding=1,
                                                          batch_norm=use_bn,
                                                          activation=activation, padding_mode=padding_mode))

            for _ in range(num_noise_feat_extractor_res):
                noise_feat_extractor.append(blocks.ResBlock(noise_feat_dim, noise_feat_dim, stride=1,
                                                            batch_norm=use_bn, activation=activation,
                                                            padding_mode=padding_mode))

            self.noise_feat_extractor = nn.Sequential(*noise_feat_extractor)

        offset_feat_extractor = []
        offset_feat_extractor.append(blocks.conv_block(2, offset_feat_dim, 3, stride=1, padding=1,
                                                       batch_norm=use_bn,
                                                       activation=activation, padding_mode=padding_mode))

        for _ in range(num_offset_feat_extractor_res):
            offset_feat_extractor.append(blocks.ResBlock(offset_feat_dim, offset_feat_dim, stride=1,
                                                         batch_norm=use_bn, activation=activation,
                                                         padding_mode=padding_mode))

        self.offset_feat_extractor = nn.Sequential(*offset_feat_extractor)

        weight_predictor = []
        weight_predictor.append(
            blocks.conv_block(project_dim * 2 + offset_feat_dim * use_offset + noise_feat_dim * use_noise_estimate,
                              2 * project_dim,
                              3, stride=1, padding=1, batch_norm=use_bn, activation=activation,
                              padding_mode=padding_mode))

        for _ in range(num_weight_predictor_res):
            weight_predictor.append(blocks.ResBlock(2 * project_dim, 2 * project_dim, stride=1,
                                                    batch_norm=use_bn, activation=activation,
                                                    padding_mode=padding_mode))
        weight_predictor.append(blocks.conv_block(2 * project_dim, input_dim, 3, stride=1, padding=1,
                                                  batch_norm=use_bn,
                                                  activation='none', padding_mode=padding_mode))

        self.weight_predictor = nn.Sequential(*weight_predictor)

    def forward(self, input_dict, noise_estimate=None):
        enc, offsets = input_dict['x'], input_dict['offsets']

        shape = enc.shape  # Batch, burst, feat, row, col

        all_feat = enc
        all_feat_proj = self.feat_project_layer(all_feat.view(-1, *all_feat.shape[-3:])).view(*all_feat.shape[:2], -1,
                                                                                              *all_feat.shape[-2:])

        if offsets is not None:
            if getattr(self, 'use_mean', False):
                base_feat_proj = all_feat_proj.mean(1, keepdim=True).repeat(1, shape[1] - 1, 1, 1, 1)
            else:
                base_feat_proj = all_feat_proj[:, :1].contiguous().repeat(1, shape[1] - 1, 1, 1, 1)
            base_feat_proj = base_feat_proj.view(-1, *base_feat_proj.shape[-3:])

            # Warp
            offsets_re = offsets.view(-1, *offsets.shape[-3:])
            base_feat_proj_warped = lispr_warp.warp(base_feat_proj, offsets_re, mode=getattr(self, 'warp_type', 'bilinear'))

            base_feat_proj_warped = base_feat_proj_warped.view(shape[0], shape[1] - 1, *base_feat_proj_warped.shape[-3:])
            base_feat_proj_warped_all = torch.cat((all_feat_proj[:, :1], base_feat_proj_warped), dim=1)
        else:
            # No motion
            if getattr(self, 'use_mean', False):
                base_feat_proj_warped_all = all_feat_proj.mean(1, keepdim=True)
            else:
                base_feat_proj_warped_all = all_feat_proj[:, :1]

        feat_diff_proj = all_feat_proj - base_feat_proj_warped_all
        feat_diff_proj = feat_diff_proj.view(-1, *feat_diff_proj.shape[-3:])
        
        if getattr(self, 'use_abs_diff', False):
            feat_diff_proj = feat_diff_proj.abs()

        all_feat_proj = all_feat_proj.view(-1, *all_feat_proj.shape[-3:])

        weight_pred_in = [all_feat_proj, feat_diff_proj]

        if getattr(self, 'use_offset', True):
            if getattr(self, 'ref_offset_noise', 0.0) > 0.0:
                offsets_base = torch.rand((shape[0], 1, 2, *shape[-2:]), device=enc.device).float() * 2 * \
                               getattr(self, 'ref_offset_noise', 0.0) - getattr(self, 'ref_offset_noise', 0.0)
            else:
                offsets_base = torch.zeros((shape[0], 1, 2, *shape[-2:]), device=enc.device).float()

            offsets_all = torch.cat((offsets_base, offsets), dim=1)
            offsets_all = offsets_all.view(-1, *offsets_all.shape[-3:])

            if getattr(self, 'offset_modulo', None) is not None:
                offsets_all = offsets_all % self.offset_modulo

            offsets_feat = self.offset_feat_extractor(offsets_all)
            weight_pred_in.append(offsets_feat)

        if getattr(self, 'use_noise_estimate', False):
            noise_estimate = noise_estimate.view(-1, *noise_estimate.shape[-3:])

            if noise_estimate.shape[-1] != all_feat_proj.shape[-1]:
                noise_estimate = F.interpolate(noise_estimate, size=(all_feat_proj.shape[-2], all_feat_proj.shape[-1]),
                                               mode='bilinear', align_corners=False)
            noise_feat = self.noise_feat_extractor(noise_estimate)
            weight_pred_in.append(noise_feat)

        weight_pred_in = torch.cat(weight_pred_in, dim=1)
        weights = self.weight_predictor(weight_pred_in)

        weights = weights.view(shape[0], -1, *weights.shape[-3:])

        if self.use_softmax:
            weights_norm = F.softmax(weights, dim=1) * shape[1]
        else:
            weights_norm = weights.abs()

        return weights_norm


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input_dict, noise_estimate=None):
        enc, offsets = input_dict['x'], input_dict['offsets']
        return torch.ones_like(enc)


