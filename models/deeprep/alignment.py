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
from models.alignment.pwcnet import PWCNet
from admin.environment import env_settings
import torch.nn.functional as F


class PWCNetAlignment(nn.Module):
    def __init__(self, train_alignmentnet=False, flow_ds_factor=1.0):
        super().__init__()
        alignment_net = PWCNet(load_pretrained=True,
                               weights_path='{}/pwcnet-network-default.pth'.format(env_settings().pretrained_nets_dir))
        self.alignment_net = alignment_net
        self.train_alignmentnet = train_alignmentnet

        self.flow_ds_factor = flow_ds_factor

    def forward(self, im, reference_id=0):
        assert im.dim() == 5

        if im.shape[2] == 4:
            im_rgb = torch.stack((im[:, :, 0], im[:, :, 1:3].mean(dim=2), im[:, :, 3]), dim=2)
        elif im.shape[2] == 3:
            im_rgb = im
        elif im.shape[2] == 1:
            im_rgb = im.repeat(1, 1, 3, 1, 1)
        else:
            raise Exception

        assert reference_id == 0

        im_ref = im_rgb[:, :1, ...].repeat(1, im_rgb.shape[1] - 1, 1, 1, 1)
        im_oth = im_rgb[:, 1:, ...]

        im_oth = im_oth.contiguous()
        im_ref = im_ref.contiguous()
        if self.train_alignmentnet:
            offsets = self.alignment_net(im_ref.view(-1, *im_ref.shape[-3:]), im_oth.view(-1, *im_oth.shape[-3:]))
        else:
            with torch.no_grad():
                self.alignment_net = self.alignment_net.eval()
                offsets = self.alignment_net(im_ref.view(-1, *im_ref.shape[-3:]), im_oth.view(-1, *im_oth.shape[-3:]))

        if self.flow_ds_factor > 1.0:
            offsets = F.interpolate(offsets, size=None, scale_factor=1.0 / self.flow_ds_factor,
                                    mode='bilinear', align_corners=False) * (1.0 / self.flow_ds_factor)
        return offsets.view(im.shape[0], -1, *offsets.shape[-3:]), {}


class AlignmentWrapper(nn.Module):
    def __init__(self, alignment_net, train_alignmentnet=False, force_3ch_input=False):
        super().__init__()
        self.alignment_net = alignment_net
        self.train_alignmentnet = train_alignmentnet
        self.force_3ch_input = force_3ch_input

    def forward(self, im, reference_id=0):
        assert im.dim() == 5

        if self.force_3ch_input:
            if im.shape[2] == 4:
                im_rgb = torch.stack((im[:, :, 0], im[:, :, 1:3].mean(dim=2), im[:, :, 3]), dim=2)
            elif im.shape[2] == 3:
                im_rgb = im
            elif im.shape[2] == 1:
                im_rgb = im.repeat(1, 1, 3, 1, 1)
            else:
                raise Exception
        else:
            im_rgb = im

        im_ref = im_rgb[:, :1, ...].repeat(1, im_rgb.shape[1] - 1, 1, 1, 1)
        im_oth = im_rgb[:, 1:, ...]

        im_oth = im_oth.contiguous()
        im_ref = im_ref.contiguous()
        if self.train_alignmentnet:
            out = self.alignment_net(im_ref.view(-1, *im_ref.shape[-3:]), im_oth.view(-1, *im_oth.shape[-3:]))
        else:
            with torch.no_grad():
                self.alignment_net = self.alignment_net.eval()
                out = self.alignment_net(im_ref.view(-1, *im_ref.shape[-3:]), im_oth.view(-1, *im_oth.shape[-3:]))

        if isinstance(out, (tuple, list)):
            offsets = out[0]
            aux_info = out[1]
        else:
            offsets = out
            aux_info = {}
        return offsets.view(im.shape[0], -1, *offsets.shape[-3:]), aux_info
