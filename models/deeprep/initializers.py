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
import torch.nn.functional as F
from models.layers.upsampling import PixShuffleUpsampler
import models.layers.blocks as blocks
import models.layers.blocks as lispr_blocks
import torch


class PixelShuffleInitializer(nn.Module):
    def __init__(self, input_dim, output_dim, upsample_factor, num_res_blocks=None, use_bn=False, activation='relu'):
        super().__init__()
        if num_res_blocks is not None:
            res_layers = []

            for _ in range(num_res_blocks):
                res_layers.append(blocks.ResBlock(input_dim, input_dim, stride=1, batch_norm=use_bn, activation=activation))
            self.res_layers = nn.Sequential(*res_layers)
        else:
            self.res_layers = None

        self.upsample_layer = PixShuffleUpsampler(input_dim, output_dim, upsample_factor=upsample_factor,
                                                  use_bn=False, activation='relu', icnrinit=True, gauss_blur_sd=None)

    def forward(self, x):
        assert x.dim() == 5

        x_ref = x[:, 0].contiguous()

        if getattr(self, 'res_layers', None) is not None:
            x_ref = self.res_layers(x_ref)

        x_ref_up = self.upsample_layer(x_ref)
        return x_ref_up


class Conv(nn.Module):
    def __init__(self, input_dim, out_dim, ksz=3, padding_mode='zeros', use_mean=False):
        super().__init__()
        layers = []

        d_in = input_dim

        layers.append(lispr_blocks.conv_block(d_in, out_dim, kernel_size=ksz, stride=1, bias=False, padding=ksz // 2,
                                              batch_norm=False, activation='none', padding_mode=padding_mode))
        self.layers = nn.Sequential(*layers)
        self.use_mean = use_mean

    def forward(self, x):
        assert x.dim() == 5

        if self.use_mean:
            shape = x.shape
            x = x.view(-1, *x.shape[-3:])
            out = self.layers(x)
            out = out.view(shape[0], shape[1], *out.shape[-3:])
            out = out.mean(1)
        else:
            x_ref = x[:, 0].contiguous()
            out = self.layers(x_ref)
        return out


class Zeros(nn.Module):
    def __init__(self, output_dim, upsample_factor):
        super().__init__()
        self.output_dim = output_dim
        self.upsample_factor = upsample_factor

    def forward(self, x):
        assert x.dim() == 5

        x_ref_up = torch.zeros((x.shape[0], self.output_dim, x.shape[-2] * self.upsample_factor,
                                x.shape[-1] * self.upsample_factor), dtype=x.dtype, device=x.device)
        return x_ref_up
