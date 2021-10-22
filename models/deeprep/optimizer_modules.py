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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.layers.warp as lispr_warp

from models.deeprep.backward_warp import backward_warp


class SteepestDescentOptimizer(nn.Module):
    def __init__(self, degradation_operator, weight_predictor, num_iter=1, compute_losses=False, detach_length=float('Inf'),
                 regularization_layer=None, init_feat_reg_w=1.0, use_feature_regularization=False):
        super().__init__()
        self.degradation_operator = degradation_operator
        self.num_iter = num_iter
        self.compute_losses = compute_losses
        self.detach_length = detach_length
        self.regularization_layer = regularization_layer
        self.weight_predictor = weight_predictor

        self.use_feature_regularization = use_feature_regularization
        self.feat_reg_w = nn.Parameter(init_feat_reg_w * torch.ones(1))

    def _sqr_norm(self, x):
        sqr_norm = (x * x).sum(list(range(1, x.ndim)))
        return sqr_norm

    def _compute_loss(self, data_residual, reg_term=None):
        loss = (data_residual ** 2).mean()

        if reg_term is not None:
            loss = loss + (reg_term ** 2).mean()
        return loss

    def warp_y(self, y, offsets):
        assert y.dim() == 5
        shape = offsets.shape
        burst_sz = shape[1] + 1

        # Warp image
        y_ref = y[:, :1, ...].contiguous()
        y_oth = y[:, 1:, ...].contiguous()

        y_oth = y_oth.view(-1, *y_oth.shape[-3:])
        offsets = offsets.view(-1, *offsets.shape[-3:])

        factor = y_oth.shape[-1] / offsets.shape[-1]
        if factor != 1:
            offsets = F.interpolate(offsets, size=None, scale_factor=factor) * factor

        self.offsets_ = offsets
        y_oth = lispr_warp.warp(y_oth, offsets, mode=getattr(self, 'warp_type', 'bilinear'))

        y_oth = y_oth.view(shape[0], burst_sz - 1, *y_oth.shape[-3:])
        y_warped = torch.cat((y_ref, y_oth), dim=1)
        y_warped = y_warped.view(-1, *y_warped.shape[-3:])

        return y_warped.view(shape[0], burst_sz, *y_warped.shape[-3:])

    def compute_residual(self, y_ds_warped_mosaicked, x, weights_norm):
        data_residual = weights_norm * (y_ds_warped_mosaicked - x)
        return data_residual

    def forward(self, y_init, x, offsets, num_iter=None, noise_estimate=None):
        num_iter = self.num_iter if num_iter is None else num_iter

        losses = []

        burst_sz = x.shape[1]
        # Obtain certainty weights
        if num_iter > 0:
            weights = self.weight_predictor({'x': x, 'offsets': offsets}, noise_estimate=noise_estimate)

        y_current = y_init.clone()

        torch_grad_enabled = torch.is_grad_enabled()
        if not torch_grad_enabled:
            self.compute_losses = False

        for i in range(num_iter):
            if (self.detach_length == 0) or (i > 0 and i % self.detach_length == 0):
                y_current = y_current.detach()

            torch.set_grad_enabled(True)
            y_current.requires_grad_(True)

            # ************************************ Forward ***************************************************
            # Perform warping
            y = y_current.unsqueeze(1).expand(-1, burst_sz, -1, -1, -1)
            y_warped = self.warp_y(y, offsets)

            torch.set_grad_enabled(torch_grad_enabled)

            # Apply feature degradation
            y_warped_ds = self.degradation_operator(y_warped)

            # Compute data loss
            data_residual = self.compute_residual(y_warped_ds, x, weights)

            # Compute regularization loss
            feat_reg_w = 0.0
            reg_term = None
            if self.use_feature_regularization:
                feat_reg_w = self.feat_reg_w / 8.0
                reg_term = feat_reg_w.abs().sqrt() * (y_current - y_init)

            # Compute total loss
            if self.compute_losses:
                losses.append(self._compute_loss(data_residual, reg_term))

            # Compute gradient w.r.t. warped image
            grad_y_warped = self.degradation_operator.apply_transposed(data_residual * weights)

            # Compute gradient w.r.t. original image
            grad_y = backward_warp(y_warped, y, grad_y_warped, offsets)[0]
            grad_y_current = grad_y.sum(dim=1)

            # add gradient due to regularization term
            if self.use_feature_regularization:
                grad_y_current_reg = reg_term.clone() * feat_reg_w.abs().sqrt()
                grad_y_current += grad_y_current_reg

            # Compute the denominator
            h = weights * self.degradation_operator(
                self.warp_y(grad_y_current.unsqueeze(1).expand(-1, burst_sz, -1, -1, -1), offsets))

            # Compute squared norms
            ip_gg = self._sqr_norm(grad_y_current)
            ip_hh = self._sqr_norm(h)

            if self.use_feature_regularization:
                ip_hh = ip_hh + ip_gg * feat_reg_w

            # Compute step length
            alpha = ip_gg / (ip_hh).clamp(1e-8)

            # Compute optimization step
            step = grad_y_current * alpha.view(*[-1 if d == 0 else 1 for d in range(grad_y_current.dim())])

            # Add step to parameter
            y_current = y_current - step

        if self.compute_losses:
            y = y_current.unsqueeze(1).expand(-1, burst_sz, -1, -1, -1)
            y_warped = self.warp_y(y, offsets)

            y_warped_ds = self.degradation_operator(y_warped)
            data_residual = self.compute_residual(y_warped_ds, x, weights)

            reg_term = None
            if self.use_feature_regularization:
                feat_reg_w = self.feat_reg_w / burst_sz
                reg_term = feat_reg_w.abs().sqrt() * (y_current - y_init)
            losses.append(self._compute_loss(data_residual, reg_term))

        # Reset the grad enabled flag
        torch.set_grad_enabled(torch_grad_enabled)
        if not torch_grad_enabled:
            y_current.detach_()

        return y_current, losses
