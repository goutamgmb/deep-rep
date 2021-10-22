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

try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except:
    print("Cannot load SpatialCorrelationSampler module")


class CostVolume(nn.Module):
    def __init__(self, kernel_size, max_displacement, stride=1):
        super().__init__()
        self.correlation_layer = SpatialCorrelationSampler(kernel_size, 2*max_displacement + 1, stride,
                                                           int((kernel_size-1)/2))

    def forward(self, feat1, feat2):
        # feat1 - b, c, h1, w1
        # feat2 - b, c, h2, w2
        # out --> b, 2*max_displacement + 1, h1, w1
        assert feat1.dim() == 4 and feat2.dim() == 4, 'Expect 4 dimensional inputs'

        batch_size = feat1.shape[0]

        cost_volume = self.correlation_layer(feat1.contiguous(), feat2.contiguous())

        return cost_volume.view(batch_size, -1, cost_volume.shape[-2], cost_volume.shape[-1])
