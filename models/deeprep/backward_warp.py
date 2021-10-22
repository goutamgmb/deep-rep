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
import models.layers.warp as lispr_warp
from torch.autograd import Function


class BackwardWarp(Function):
    @staticmethod
    def forward(ctx, y_warped, y, grad_y_warped, offsets):
        """ backward pass through the warping operation """
        # y_ds = y_ds.unsqueeze(1).expand(-1, y_ds_warped.shape[1], -1, -1, -1)
        grad_y = torch.autograd.grad(y_warped, y, grad_y_warped, create_graph=True, retain_graph=True)
        ctx.save_for_backward(y_warped, y, grad_y_warped, offsets, grad_y[0])
        return grad_y

    @staticmethod
    def backward(ctx, grad_output):
        """ backward of the backward pass through the warping operation """
        y_warped, y, grad_y_warped, offsets, grad_y = ctx.saved_tensors

        grad_y_warped = None
        grad_y = None
        grad_offset = None

        shape = offsets.shape
        burst_sz = shape[1] + 1

        grad_output = grad_output.view(shape[0], burst_sz, *grad_output.shape[-3:])
        # Warp image
        reference_id = 0
        grad_output_ref = grad_output[:, reference_id:reference_id + 1, ...]
        grad_output_oth = grad_output[:, list(range(reference_id)) + list(range(reference_id + 1, burst_sz)), ...]

        grad_output_oth = grad_output_oth.view(-1, *grad_output_oth.shape[-3:])
        offsets = offsets.view(-1, *offsets.shape[-3:])

        factor = grad_output_oth.shape[-1] / offsets.shape[-1]
        if factor != 1:
            offsets = F.interpolate(offsets, size=None, scale_factor=factor) * factor

        grad_output_oth = lispr_warp.warp(grad_output_oth, offsets, 'bilinear')
        grad_output_oth = grad_output_oth.view(shape[0], burst_sz - 1, *grad_output_oth.shape[-3:])
        grad_grad_y_warped = torch.cat((grad_output_ref, grad_output_oth), dim=1)
        grad_grad_y_warped = grad_grad_y_warped.view(shape[0], burst_sz, *grad_grad_y_warped.shape[-3:])

        return (grad_y_warped, grad_y, grad_grad_y_warped, grad_offset)


backward_warp = BackwardWarp.apply
