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
from dataset.base_image_dataset import BaseImageDataset
from data.image_loader import jpeg4py_loader
from admin.environment import env_settings


class OpenImagesDataset(BaseImageDataset):
    """
    Dataset class for loading the OpenImages dataset [1]. The dataset can be downlaoded from
    https://storage.googleapis.com/openimages/web/download.html

    [1] Openimages: A public dataset for large-scale multilabel and multi-class image classification.
        Ivan Krasin, Tom Duerig, Neil Alldrin, Vittorio Ferrari, Sami Abu-El-Haija, Alina Kuznetsova, Hassan Rom,
        Jasper Uijlings, Stefan Popov, Andreas Veit, Serge Belongie, Victor Gomes, Abhinav Gupta, Chen Sun, Gal Chechik,
        David Cai, Zheyun Feng, Dhyanesh Narayanan, and Kevin Murphy
    """
    def __init__(self, root=None, split='train', image_loader=jpeg4py_loader, initialize=True):
        """
        args:
            root - Path to root dataset directory
            split - Dataset split to use. Can be 'train' or 'test'
            image_loader - loader used to read the images
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        root = env_settings().openimages_dir if root is None else root
        super().__init__('OpenImages', root, image_loader)
        self.split = split

        if initialize:
            self.initialize()

    def initialize(self):
        root = self.root
        split = self.split
        self.img_pth = root + '/' + split
        self.image_list = self._get_image_list()

    def _get_image_list(self):
        image_list = sorted(os.listdir(self.img_pth))

        return image_list

    def get_image_info(self, im_id):
        return {}

    def _get_image(self, im_id):
        path = os.path.join(self.img_pth, self.image_list[im_id])
        img = self.image_loader(path)
        return img

    def get_image(self, im_id, info=None):
        frame = self._get_image(im_id)

        if info is None:
            info = self.get_image_info(im_id)

        return frame, info
