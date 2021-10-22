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
import models.deeprep.initializers as initializers
import models.deeprep.encoders as encoders
import models.deeprep.degradation_layers as ds_layers
import models.deeprep.optimizer_modules as optimizer_modules
import models.deeprep.decoders as decoders
import models.deeprep.weight_predictors as weightpredictors
from admin.model_constructor import model_constructor
from models.alignment.pyrcorr import PyrCorr

from models.deeprep.alignment import PWCNetAlignment, AlignmentWrapper


class DeepRepNet(nn.Module):
    """ Deep Reparametrization model"""
    def __init__(self, lr_encoder, hr_decoder, hr_initializer, optimizer,
                 alignment_net=None, use_noise_estimate=False):
        super().__init__()
        self.lr_encoder = lr_encoder        # Encoder
        self.hr_decoder = hr_decoder        # Decoder

        self.hr_initializer = hr_initializer    # Initializer module which predicts initial latent encoding of output image
        self.optimizer = optimizer              # Optimizer module which minimizes the feature space reconstruction error
        self.alignment_net = alignment_net      # Network which computes alignment vectors

        self.use_noise_estimate = use_noise_estimate

    def forward(self, im, num_iter=None, noise_estimate=None):
        enc_in = im
        if noise_estimate is not None and self.use_noise_estimate:
            if noise_estimate.dim() == 4:
                noise_estimate_re = noise_estimate.unsqueeze(1).repeat(1, im.shape[1], 1, 1, 1)
            else:
                noise_estimate_re = noise_estimate

            enc_in = torch.cat((im, noise_estimate_re), dim=2)
        else:
            noise_estimate_re = None

        lr_enc = self.lr_encoder(enc_in)

        y_init = self.hr_initializer(lr_enc['enc'])

        if self.alignment_net is not None:
            offsets, offsets_aux = self.alignment_net(im)
        else:
            offsets = None
            offsets_aux = {}
        y_optim, _ = self.optimizer(y_init=y_init, x=lr_enc['enc'], offsets=offsets, num_iter=num_iter,
                                    noise_estimate=noise_estimate_re)

        lr_enc['fused_enc'] = y_optim
        y_pred = self.hr_decoder(lr_enc)
        return y_pred['pred'], {'offsets': offsets, 'offsets_aux': offsets_aux}


@model_constructor
def deeprep_sr_iccv21(num_iter, enc_dim=32, enc_num_res_blocks=3, enc_out_dim=64,
                      dec_dim_pre=64, dec_dim_post=32, dec_num_pre_res_blocks=3, dec_num_post_res_blocks=3,
                      dec_in_dim=32, dec_upsample_factor=2,
                      feature_degradation_upsample_factor=2,
                      wp_project_dim=64,
                      wp_offset_feat_dim=64,
                      wp_num_offset_feat_extractor_res=1,
                      wp_num_weight_predictor_res=3,
                      wp_ref_offset_noise=0.02,
                      use_feature_regularization=False,
                      init_feat_reg_w=1.0, gauss_blur_sd=1,
                      ):
    lr_encoder = encoders.ResEncoder(input_channels=4, init_dim=enc_dim, num_res_blocks=enc_num_res_blocks,
                                     out_dim=enc_out_dim)
    hr_decoder = decoders.ResPixShuffleConv(dec_in_dim,
                                            dec_dim_pre, dec_num_pre_res_blocks, dec_dim_post,
                                            dec_num_post_res_blocks,
                                            upsample_factor=dec_upsample_factor,
                                            icnrinit=True, gauss_blur_sd=gauss_blur_sd)

    hr_initializer = initializers.PixelShuffleInitializer(input_dim=enc_out_dim, output_dim=dec_in_dim,
                                                          upsample_factor=feature_degradation_upsample_factor)

    ds = ds_layers.Conv(input_dim=dec_in_dim, out_dim=enc_out_dim, stride=feature_degradation_upsample_factor, ksz=3)

    weight_predictor = weightpredictors.SimpleWeightPredictor(enc_out_dim, wp_project_dim, wp_offset_feat_dim,
                                                              num_offset_feat_extractor_res=wp_num_offset_feat_extractor_res,
                                                              num_weight_predictor_res=wp_num_weight_predictor_res,
                                                              use_bn=False, activation='relu',
                                                              use_noise_estimate=False,
                                                              offset_modulo=1, use_offset=True,
                                                              ref_offset_noise=wp_ref_offset_noise,
                                                              use_softmax=False)

    optimizer = optimizer_modules.SteepestDescentOptimizer(degradation_operator=ds, num_iter=num_iter,
                                                           compute_losses=False,
                                                           weight_predictor=weight_predictor,
                                                           use_feature_regularization=use_feature_regularization,
                                                           init_feat_reg_w=init_feat_reg_w)

    alignment_net = PWCNetAlignment()
    net = DeepRepNet(lr_encoder=lr_encoder, hr_decoder=hr_decoder, hr_initializer=hr_initializer,
                     optimizer=optimizer, alignment_net=alignment_net)
    return net



@model_constructor
def deeprep_denoise_iccv21(num_iter, enc_dim=32, enc_num_res_blocks=3, enc_out_dim=64,
                           dec_dim_pre=64, dec_num_res_blocks=3,
                           dec_in_dim=32,
                           wp_project_dim=64,
                           wp_offset_feat_dim=64,
                           wp_num_offset_feat_extractor_res=1,
                           wp_num_weight_predictor_res=3,
                           wp_ref_offset_noise=0.02, use_softmax=False,
                           use_feature_regularization=False,
                           init_feat_reg_w=1.0,
                           wp_use_abs_diff=False,
                           train_alignmentnet=False,
                           use_noise_estimate=False,
                           wp_noise_feat_dim=32,
                           wp_num_noise_feat_extractor_res=1,
                           padding_mode='zeros', activation='relu',
                           color_input=True):
    input_channels = 3*(1 + use_noise_estimate) if color_input else (1 + use_noise_estimate)

    lr_encoder = encoders.ResEncoder(input_channels=input_channels,
                                     init_dim=enc_dim,
                                     num_res_blocks=enc_num_res_blocks,
                                     out_dim=enc_out_dim, padding_mode=padding_mode, activation=activation)

    output_channels = 3 if color_input else 1
    hr_decoder = decoders.ResDecoder(dec_in_dim, dec_dim_pre, dec_num_res_blocks, out_dim=output_channels,
                                     padding_mode=padding_mode, activation=activation)

    hr_initializer = initializers.Conv(input_dim=enc_out_dim, out_dim=dec_in_dim, ksz=3, padding_mode=padding_mode)

    ds = ds_layers.Conv(input_dim=dec_in_dim, out_dim=enc_out_dim, ksz=3, padding_mode=padding_mode)

    weight_predictor = weightpredictors.SimpleWeightPredictor(enc_out_dim, wp_project_dim, wp_offset_feat_dim,
                                                              num_offset_feat_extractor_res=wp_num_offset_feat_extractor_res,
                                                              num_weight_predictor_res=wp_num_weight_predictor_res,
                                                              use_bn=False, activation=activation,
                                                              use_noise_estimate=use_noise_estimate,
                                                              offset_modulo=1, use_offset=True,
                                                              ref_offset_noise=wp_ref_offset_noise,
                                                              use_softmax=use_softmax,
                                                              use_abs_diff=wp_use_abs_diff,
                                                              noise_feat_dim=wp_noise_feat_dim,
                                                              num_noise_feat_extractor_res=wp_num_noise_feat_extractor_res,
                                                              padding_mode=padding_mode,
                                                              color_input=color_input)

    optimizer = optimizer_modules.SteepestDescentOptimizer(degradation_operator=ds, num_iter=num_iter,
                                                           weight_predictor=weight_predictor,
                                                           use_feature_regularization=use_feature_regularization,
                                                           init_feat_reg_w=init_feat_reg_w)

    alignment_net = PWCNetAlignment(train_alignmentnet=train_alignmentnet)
    net = DeepRepNet(lr_encoder=lr_encoder, hr_decoder=hr_decoder, hr_initializer=hr_initializer,
                     optimizer=optimizer, alignment_net=alignment_net, use_noise_estimate=use_noise_estimate)
    return net


@model_constructor
def deeprep_denoise_custom_flow_iccv21(num_iter, enc_dim=32, enc_num_res_blocks=3,
                                       enc_out_dim=64,
                                       dec_dim_pre=64, dec_num_res_blocks=3,
                                       dec_in_dim=32,
                                       wp_project_dim=64,
                                       wp_offset_feat_dim=64,
                                       wp_num_offset_feat_extractor_res=1,
                                       wp_num_weight_predictor_res=3,
                                       wp_ref_offset_noise=0.02, use_softmax=False,
                                       use_feature_regularization=True,
                                       init_feat_reg_w=1.0,
                                       wp_use_abs_diff=False,
                                       use_ref_for_offset=True,
                                       use_noise_estimate=False,
                                       wp_noise_feat_dim=32,
                                       wp_num_noise_feat_extractor_res=1,
                                       padding_mode='zeros',
                                       align_init_dim=64, align_num_res_blocks=5,
                                       align_ds_factor=2,
                                       offset_cdim=64,
                                       offset_predictor_dims=(128, 64),
                                       corr_max_disp=(2, 2, 4),
                                       color_input=True
                                       ):
    input_channels = 3 * (1 + use_noise_estimate) if color_input else (1 + use_noise_estimate)

    lr_encoder = encoders.ResEncoder(input_channels=input_channels, init_dim=enc_dim,
                                     num_res_blocks=enc_num_res_blocks,
                                     out_dim=enc_out_dim, padding_mode=padding_mode)

    output_channels = 3 if color_input else 1
    hr_decoder = decoders.ResDecoder(dec_in_dim,
                                     dec_dim_pre, dec_num_res_blocks, out_dim=output_channels,
                                     padding_mode=padding_mode)

    hr_initializer = initializers.Conv(input_dim=enc_out_dim, out_dim=dec_in_dim, ksz=3, padding_mode=padding_mode)

    ds = ds_layers.Conv(input_dim=dec_in_dim, out_dim=enc_out_dim, ksz=3, padding_mode=padding_mode)

    weight_predictor = weightpredictors.SimpleWeightPredictor(enc_out_dim, wp_project_dim, wp_offset_feat_dim,
                                                              num_offset_feat_extractor_res=wp_num_offset_feat_extractor_res,
                                                              num_weight_predictor_res=wp_num_weight_predictor_res,
                                                              use_bn=False, activation='relu',
                                                              use_noise_estimate=use_noise_estimate,
                                                              offset_modulo=1, use_offset=True,
                                                              ref_offset_noise=wp_ref_offset_noise,
                                                              use_softmax=use_softmax,
                                                              use_abs_diff=wp_use_abs_diff,
                                                              noise_feat_dim=wp_noise_feat_dim,
                                                              num_noise_feat_extractor_res=wp_num_noise_feat_extractor_res,
                                                              padding_mode=padding_mode,
                                                              color_input=color_input)

    optimizer = optimizer_modules.SteepestDescentOptimizer(degradation_operator=ds, num_iter=num_iter,
                                                           weight_predictor=weight_predictor,
                                                           use_feature_regularization=use_feature_regularization,
                                                           init_feat_reg_w=init_feat_reg_w)

    flow_input_channels = 3 if color_input else 1
    pyrcorr_net = PyrCorr(flow_input_channels, align_init_dim, align_num_res_blocks, align_ds_factor, offset_cdim, offset_predictor_dims,
                          corr_max_disp=corr_max_disp, corr_kernel_sz=3, use_ref_for_offset=use_ref_for_offset,
                          rgb2bgr=False, use_bn=False, activation='relu')
    alignment_net = AlignmentWrapper(alignment_net=pyrcorr_net, train_alignmentnet=True,
                                     force_3ch_input=False)

    net = DeepRepNet(lr_encoder=lr_encoder, hr_decoder=hr_decoder, hr_initializer=hr_initializer,
                     optimizer=optimizer, alignment_net=alignment_net, use_noise_estimate=use_noise_estimate)
    return net
