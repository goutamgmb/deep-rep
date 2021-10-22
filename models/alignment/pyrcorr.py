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
import torch.nn as nn
import math
import torch.nn.functional as F
import models.layers.warp as lispr_warp
import models.layers.blocks as blocks
from models.layers.correlation import CostVolume


class PyrCorr(torch.nn.Module):
    """ Optical flow network based on PWC-Net architecture """
    def __init__(self, input_channels, init_dim, num_res_blocks, ds_factor, offset_cdim, offset_predictor_dims,
                 corr_max_disp=(2, 2, 4), corr_kernel_sz=3, use_ref_for_offset=True, rgb2bgr=False, use_bn=False,
                 activation='relu'):
        super(PyrCorr, self).__init__()
        self.rgb2bgr = rgb2bgr
        self.input_channels = input_channels

        self.ds_factor = ds_factor
        self.init_layer = blocks.conv_block(input_channels, init_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                            activation=activation)

        d_in = init_dim
        offset_res_layers = []

        remaining_ds_factor = ds_factor
        for _ in range(num_res_blocks):
            offset_res_layers.append(blocks.ResBlock(d_in, d_in, stride=1, batch_norm=use_bn, activation=activation))

            if remaining_ds_factor > 1.1:
                offset_res_layers.append(blocks.conv_block(d_in, d_in, 3, stride=2, padding=1, batch_norm=use_bn,
                                                           activation=activation))
                remaining_ds_factor = remaining_ds_factor // 2

        offset_d_in = d_in
        self.offset_res_layers = nn.Sequential(*offset_res_layers)

        offsetfeat_layers = []
        corr_layers = []
        offset_predictors = []

        for i, md in enumerate(corr_max_disp):
            stride = 1 if i == 0 else 2
            offsetfeat_layers.append(
                blocks.conv_block(offset_d_in, offset_cdim, 3, stride=stride, padding=1, batch_norm=use_bn,
                                  activation=activation))
            offset_d_in = offset_cdim
            corr_layers.append(CostVolume(corr_kernel_sz, max_displacement=md))

            d_in = (md * 2 + 1) ** 2 + offset_cdim * use_ref_for_offset
            offset_predictor = []
            for d_out in offset_predictor_dims:
                offset_predictor.append(blocks.conv_block(d_in, d_out, 3, stride=1, padding=1, batch_norm=use_bn,
                                                          activation=activation))
                d_in = d_out

            offset_predictor.append(blocks.conv_block(d_in, 2, 3, stride=1, padding=1,
                                                      batch_norm=False, activation='none'))

            offset_predictor = nn.Sequential(*offset_predictor)
            offset_predictors.append(offset_predictor)

        self.offsetfeat_layers = nn.ModuleList(offsetfeat_layers)
        self.corr_layers = nn.ModuleList(corr_layers)

        self.offset_predictors = nn.ModuleList(offset_predictors)

        self.use_ref_for_offset = use_ref_for_offset

    def estimate_flow(self, source_img, target_img):
        shape = source_img.shape

        source_img = source_img.view(-1, *source_img.shape[-3:])
        source_feat = self.init_layer(source_img)
        source_feat = self.offset_res_layers(source_feat)

        target_img = target_img.view(-1, *target_img.shape[-3:])
        target_feat = self.init_layer(target_img)
        target_feat = self.offset_res_layers(target_feat)

        source_offsetfeat_all = []
        target_offsetfeat_all = []

        source_feat_in = source_feat
        target_feat_in = target_feat

        for l in self.offsetfeat_layers:
            source_feat_in = l(source_feat_in)
            target_feat_in = l(target_feat_in)

            source_offsetfeat_all.append(source_feat_in)
            target_offsetfeat_all.append(target_feat_in)

        prev_offsets = 0.0

        offset_preds_all = []
        for id in range(len(source_offsetfeat_all) - 1, -1, -1):
            source_offsetfeat = source_offsetfeat_all[id]
            target_offsetfeat = target_offsetfeat_all[id]

            corr = self.corr_layers[id](target_offsetfeat, source_offsetfeat)

            if getattr(self, 'use_ref_for_offset', True):
                corr_feat = torch.cat((corr, target_offsetfeat.view(-1, *target_offsetfeat.shape[-3:])), dim=1)
            else:
                corr_feat = corr
            offsets = self.offset_predictors[id](corr_feat)
            offsets = offsets + prev_offsets

            # upsample flow to account for ds factor
            offset_preds_all.append(offsets.clone())

            if id != 0:
                offsets = F.interpolate(offsets,
                                        size=(
                                            source_offsetfeat_all[id - 1].shape[-2], source_offsetfeat_all[id - 1].shape[-1]),
                                        mode='bilinear') * 2.0

                prev_offsets = offsets
                source_offsetfeat_all[id - 1] = lispr_warp.warp(source_offsetfeat_all[id - 1], offsets)

        offsets = F.interpolate(offsets, scale_factor=self.ds_factor, mode='bilinear') * self.ds_factor
        return offsets, offset_preds_all

    def forward(self, source_img, target_img):
        assert (source_img.shape[-1] == target_img.shape[-1])
        assert (source_img.shape[-2] == target_img.shape[-2])

        int_width = source_img.shape[-1]
        int_height = source_img.shape[-2]

        source_img = source_img.view(-1, *source_img.shape[-3:])
        target_img = target_img.view(-1, *target_img.shape[-3:])

        if self.rgb2bgr:
            source_img = source_img[:, [2, 1, 0]].contiguous()
            target_img = target_img[:, [2, 1, 0]].contiguous()

        int_preprocessed_width = int(math.floor(math.ceil(int_width / 64.0) * 64.0))
        int_preprocessed_height = int(math.floor(math.ceil(int_height / 64.0) * 64.0))

        # Make size multiple of 64
        source_img_re = torch.nn.functional.interpolate(input=source_img,
                                                        size=(int_preprocessed_height, int_preprocessed_width),
                                                        mode='bilinear', align_corners=False)
        target_img_re = torch.nn.functional.interpolate(input=target_img,
                                                        size=(int_preprocessed_height, int_preprocessed_width),
                                                        mode='bilinear', align_corners=False)

        flow, flow_all = self.estimate_flow(target_img_re, source_img_re)

        flow = 20.0 * torch.nn.functional.interpolate(input=flow, size=(int_height, int_width), mode='bilinear',
                                                      align_corners=False)

        if not (int_preprocessed_width == int_width and int_preprocessed_height == int_height):
            flow_all = None
        else:
            flow_all = [20.0 * f for f in flow_all]

        scale_factor_x = float(int_width) / float(int_preprocessed_width)
        scale_factor_y = float(int_height) / float(int_preprocessed_height)
        flow = torch.stack((flow[:, 0] * scale_factor_x, flow[:, 1] * scale_factor_y), dim=1)

        return flow, {'flow_all': flow_all}