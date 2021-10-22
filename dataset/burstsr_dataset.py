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

import os
import torch
import cv2
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import zipfile
import shutil
from dataset.base_rawburst_dataset import BaseRawBurstDataset
from admin.environment import env_settings
from data import processing, sampler


def load_txt(path):
    with open(path, 'r') as fh:
        out = [d.rstrip() for d in fh.readlines()]

    return out


class SamsungRAWImage:
    """ Custom class for RAW images captured from Samsung Galaxy S8 """
    @staticmethod
    def load(path):
        im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

        return SamsungRAWImage(im_raw, meta_data['black_level'], meta_data['cam_wb'],
                               meta_data['daylight_wb'], meta_data['color_matrix'], meta_data['exif_data'],
                               meta_data.get('im_preview', None))

    def __init__(self, im_raw, black_level, cam_wb, daylight_wb, color_matrix, exif_data, im_preview=None):
        self.im_raw = im_raw
        self.black_level = black_level
        self.cam_wb = cam_wb
        self.daylight_wb = daylight_wb
        self.color_matrix = color_matrix
        self.exif_data = exif_data
        self.im_preview = im_preview

        self.norm_factor = 1023.0

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'color_matrix': self.color_matrix.tolist()}

    def get_exposure_time(self):
        return self.exif_data['Image ExposureTime'].values[0].decimal()

    def get_noise_profile(self):
        noise = self.exif_data['Image Tag 0xC761'].values
        noise = [n[0] for n in noise]
        noise = np.array(noise).reshape(3, 2)
        return noise

    def get_f_number(self):
        return self.exif_data['Image FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['Image ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(4, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(4, 1, 1)

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def shape(self):
        shape = (4, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]

        if self.im_preview is not None:
            im_preview = self.im_preview[2*r1:2*r2, 2*c1:2*c2]
        else:
            im_preview = None

        return SamsungRAWImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.color_matrix,
                               self.exif_data, im_preview=im_preview)

    def postprocess(self, return_np=True, norm_factor=None):
        raise NotImplementedError


class CanonImage:
    """ Custom class for RAW images captured from Canon DSLR """
    @staticmethod
    def load(path):
        im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

        return CanonImage(im_raw.float(), meta_data['black_level'], meta_data['cam_wb'],
                          meta_data['daylight_wb'], meta_data['rgb_xyz_matrix'], meta_data['exif_data'])

    @staticmethod
    def generate_processed_image(im, meta_data, return_np=False, external_norm_factor=None, gamma=True, smoothstep=True,
                                 no_white_balance=False):
        im = im * meta_data.get('norm_factor', 1.0)

        if not meta_data.get('black_level_subtracted', False):
            im = (im - torch.tensor(meta_data['black_level'])[[0, 1, -1]].view(3, 1, 1))

        if not meta_data.get('while_balance_applied', False) and not no_white_balance:
            im = im * torch.tensor(meta_data['cam_wb'])[[0, 1, -1]].view(3, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]

        im_out = im

        if external_norm_factor is None:
            im_out = im_out / (im_out.mean() * 5.0)
        else:
            im_out = im_out / external_norm_factor

        im_out = im_out.clamp(0.0, 1.0)

        if gamma:
            im_out = im_out ** (1.0 / 2.2)

        if smoothstep:
            # Smooth curve
            im_out = 3 * im_out ** 2 - 2 * im_out ** 3

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)
        return im_out

    def __init__(self, im_raw, black_level, cam_wb, daylight_wb, rgb_xyz_matrix, exif_data):
        super(CanonImage, self).__init__()
        self.im_raw = im_raw

        if len(black_level) == 4:
            black_level = [black_level[0], black_level[1], black_level[3]]
        self.black_level = black_level

        if len(cam_wb) == 4:
            cam_wb = [cam_wb[0], cam_wb[1], cam_wb[3]]
        self.cam_wb = cam_wb

        if len(daylight_wb) == 4:
            daylight_wb = [daylight_wb[0], daylight_wb[1], daylight_wb[3]]
        self.daylight_wb = daylight_wb

        self.rgb_xyz_matrix = rgb_xyz_matrix

        self.exif_data = exif_data

        self.norm_factor = 16383

    def shape(self):
        shape = (3, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'rgb_xyz_matrix': self.rgb_xyz_matrix.tolist(), 'norm_factor': self.norm_factor}

    def get_exposure_time(self):
        return self.exif_data['EXIF ExposureTime'].values[0].decimal()

    def get_f_number(self):
        return self.exif_data['EXIF FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['EXIF ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(3, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(3, 1, 1) / 1024.0

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def set_image_data(self, im_data):
        self.im_raw = im_data

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]
        return CanonImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.rgb_xyz_matrix,
                          self.exif_data)

    def set_crop_info(self, crop_info):
        self.crop_info = crop_info

    def resize(self, size=None, scale_factor=None):
        self.im_raw = F.interpolate(self.im_raw.unsqueeze(0), size=size, scale_factor=scale_factor,
                                    mode='bilinear').squeeze(0)

    def postprocess(self, return_np=True):
        raise NotImplementedError


class BurstSRDataset(BaseRawBurstDataset):
    """
    Real-world burst super-resolution dataset. introduced in [1]. The dataset can be downloaded using
    util_scripts/download_burstsr_dataset.py script

    [1] Deep Burst Super-Resolution. Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu Timofte. CVPR 2021
    """
    def __init__(self, root=None, split='train', seq_ids=None, initialize=True):
        """
        args:
            root - Path to root dataset directory
            split - Can be 'train' or 'val'
            seq_ids - (Optional) List of sequences to load. If specified, the 'split' argument is ignored.
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        root = env_settings().burstsr_dir if root is None else root
        super().__init__('BurstSRDataset', root)

        self.split = split
        self.seq_ids = seq_ids
        self.root = root

        if initialize:
            self.initialize()

        self.initialized = initialize

    def initialize(self):
        split = self.split
        seq_ids = self.seq_ids
        self.burst_list = self._get_burst_list(split, seq_ids)

    def _get_burst_list(self, split, seq_ids):
        burst_list = sorted(os.listdir('{}/{}'.format(self.root, self.split)))
        if split is None and seq_ids is not None:
            burst_list = [b for b in burst_list if b[:4] in seq_ids]
        elif split is not None:
            lispr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            seq_ids = load_txt('{}/data_specs/burstsr_{}.txt'.format(lispr_path, split))
            burst_list = [b for b in burst_list if b[:4] in seq_ids]

        return burst_list

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def _get_raw_image(self, burst_id, im_id):
        raw_image = SamsungRAWImage.load('{}/{}/{}/samsung_{:02d}'.format(self.root, self.split,
                                                                          self.burst_list[burst_id], im_id))
        return raw_image

    def _get_gt_image(self, burst_id):
        canon_im = CanonImage.load('{}/{}/{}/canon'.format(self.root, self.split, self.burst_list[burst_id]))
        return canon_im

    def get_burst(self, burst_id, im_ids, info=None):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]

        gt = self._get_gt_image(burst_id)
        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, gt, info


def get_burstsr_val_set():
    """ Get the BurstSR validation dataset """
    burstsr_dataset = BurstSRDataset(split='val', initialize=True)
    processing_fn = processing.BurstSRProcessing(transform=None, random_flip=False,
                                                 substract_black_level=True,
                                                 crop_sz=80)
    # Train sampler and loader
    dataset = sampler.IndexedBurst(burstsr_dataset, burst_size=14, processing=processing_fn)
    return dataset
