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

import torch.optim as optim
import dataset as datasets
from data import processing, sampler, DataLoader
from trainers import SimpleTrainer
from admin.multigpu import MultiGPU
from models.loss.image_quality_v2 import PSNR, PixelWiseError
from utils.loading import load_network
from models.alignment.pwcnet import PWCNet
from admin.environment import env_settings
import actors.deeprep_actors as deeprep_actors


def run(settings):
    settings.description = 'Default parameters for training Deep Reparametrization model for RAW burst super-resolution' \
                           'on real-world BurstSR dataset'
    settings.batch_size = 16
    settings.num_workers = 8
    settings.multi_gpu = True
    settings.print_interval = 1

    settings.crop_sz = 56
    settings.burst_sz = 14

    burstsr_train = datasets.BurstSRDataset(split='train')
    burstsr_val = datasets.BurstSRDataset(split='val')

    data_processing_train = processing.BurstSRProcessing(transform=None, random_flip=True,
                                                         substract_black_level=True,
                                                         crop_sz=settings.crop_sz)
    data_processing_val = processing.BurstSRProcessing(transform=None,
                                                       substract_black_level=True, crop_sz=settings.crop_sz)

    dataset_train = sampler.RandomBurst([burstsr_train], [1], burst_size=settings.burst_sz,
                                        samples_per_epoch=settings.batch_size * 1000, processing=data_processing_train,
                                        random_reference_image=False)
    dataset_val = sampler.IndexedBurst([burstsr_val], burst_size=settings.burst_sz, processing=data_processing_val,
                                       random_reference_image=False)

    loader_train = DataLoader('train', dataset_train, training=True, num_workers=settings.num_workers,
                              stack_dim=0, batch_size=settings.batch_size)

    loader_val = DataLoader('val', dataset_val, training=False, num_workers=settings.num_workers,
                            stack_dim=0, batch_size=settings.batch_size)

    net = load_network('deeprep/sr_synthetic_default')

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=0)

    objective = {
        'rgb':  PixelWiseError(metric='l1', boundary_ignore=40),
        'psnr': PSNR(boundary_ignore=40),
    }

    loss_weight = {
        'rgb': 10.0,
    }

    pwcnet = PWCNet(load_pretrained=True,
                    weights_path='{}/pwcnet-network-default.pth'.format(env_settings().pretrained_nets_dir))

    actor = deeprep_actors.DeepRepSRBurstSRActor(net=net, objective=objective, loss_weight=loss_weight,
                                                 alignment_net=pwcnet)

    optimizer = optim.Adam([{'params': actor.net.parameters(), 'lr': 1e-4}],
                           lr=2e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = SimpleTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(40, load_latest=True, fail_safe=True)
