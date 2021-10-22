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
import models.layers.blocks as blocks


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {'enc': x.contiguous()}


class ResEncoder(nn.Module):
    def __init__(self, input_channels, init_dim, num_res_blocks, out_dim, use_bn=False, activation='relu',
                 init_stride=1, padding_mode='zeros'):
        super().__init__()

        if init_stride == 1:
            self.init_layer = blocks.conv_block(input_channels, init_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                                activation=activation, padding_mode=padding_mode)
        else:
            self.init_layer = nn.Sequential(blocks.conv_block(input_channels, init_dim, 3,
                                                              stride=1, padding=1, batch_norm=use_bn,
                                                              activation=activation, padding_mode=padding_mode),
                                            blocks.conv_block(init_dim, init_dim, 3,
                                                              stride=init_stride, padding=1, batch_norm=use_bn,
                                                              activation=activation, padding_mode=padding_mode)
                                            )
        d_in = init_dim
        res_layers = []

        for _ in range(num_res_blocks):
            res_layers.append(blocks.ResBlock(d_in, d_in, stride=1, batch_norm=use_bn, activation=activation,
                                              padding_mode=padding_mode))
        self.res_layers = nn.Sequential(*res_layers)

        self.out_layer = blocks.conv_block(d_in, out_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                           activation=activation,
                                           padding_mode=padding_mode)

    def forward(self, x):
        burst_mode = False
        if x.dim() == 5:
            shape = x.shape
            x = x.view(-1, *x.shape[-3:])
            burst_mode = True

        out = self.init_layer(x)
        out = self.res_layers(out)
        out = self.out_layer(out)

        if burst_mode:
            out = out.view(shape[0], shape[1], *out.shape[-3:])

        return {'enc': out}
