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
import torch.nn.functional as F
import math
import models.layers.blocks as lispr_blocks


class Conv(nn.Module):
    def __init__(self, input_dim, out_dim, ksz=3, padding_mode='zeros', stride=1):
        super().__init__()
        self.ksz = ksz
        self.stride = stride
        layers = []

        d_in = input_dim

        layers.append(lispr_blocks.conv_block(d_in, out_dim, kernel_size=ksz, stride=stride, bias=False, padding=ksz // 2,
                                              batch_norm=False, activation='none', padding_mode=padding_mode))
        self.ds = nn.Sequential(*layers)

    def forward(self, x):
        x = x.contiguous()
        burst_sz = None
        if x.dim() == 5:
            burst_sz = x.shape[1]
            x = x.view(-1, *x.shape[-3:])

        x_ds = self.ds(x)
        if burst_sz is not None:
            x_ds = x_ds.view(-1, burst_sz, *x_ds.shape[-3:])
        return x_ds

    def apply_transposed(self, grad_output):
        burst_sz = None
        if grad_output.dim() == 5:
            burst_sz = grad_output.shape[1]
            grad_output = grad_output.view(-1, *grad_output.shape[-3:])

        ds = self.ds[0][0]

        stride = getattr(self, 'stride', 1)
        output_padding = stride - 1
        grad_in = F.conv_transpose2d(grad_output, ds.weight, ds.bias,
                                     stride=stride, padding=self.ksz // 2,
                                     output_padding=output_padding)

        if burst_sz is not None:
            grad_in = grad_in.view(-1, burst_sz, *grad_in.shape[-3:])

        return grad_in
