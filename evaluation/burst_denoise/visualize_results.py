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
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from utils.opencv_plotting import BurstSRVis
import torch
from models.loss.image_quality_v2 import PSNR
import cv2
import numpy as np
import argparse
import importlib
from data.postprocessing_functions import DenoisingPostProcess
from dataset.grayscale_denoise_test_set import GrayscaleDenoiseTestSet
from dataset.color_denoise_test_set import ColorDenoiseTestSet
from admin.environment import env_settings


def visualize_results(setting_name, noise_level=2, mode='grayscale'):
    """ Visualize the results on the Denoising test sets. setting_name denotes
        the name of the experiment setting, which contains the list of methods for which to visualize results.
    """

    expr_module = importlib.import_module('evaluation.burst_denoise.experiments.{}'.format(setting_name))
    expr_func = getattr(expr_module, 'main')
    network_list = expr_func()

    base_results_dir = env_settings().save_data_path
    metric = PSNR(boundary_ignore=None)
    vis = BurstSRVis(boundary_ignore=40, metric=metric)

    if mode == 'grayscale':
        dataset = GrayscaleDenoiseTestSet(noise_level=noise_level)
    else:
        dataset = ColorDenoiseTestSet(noise_level=noise_level)

    process_fn = DenoisingPostProcess(return_np=True)
    base_out_dir = '{}/denoise_{}/noise_{}'.format(base_results_dir, mode, noise_level)

    for idx in range(len(dataset)):
        burst, gt, meta_info = dataset[idx]
        burst_name = meta_info['burst_name']

        pred_all = []
        titles_all = []
        for n in network_list:
            pred_path = '{}/{}/{}.png'.format(base_out_dir, n.get_unique_name(), burst_name)
            pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)

            if mode == 'color':
                pred = torch.from_numpy(pred.astype(np.float32) / 2 ** 14).permute(2, 0, 1)
            else:
                pred = torch.from_numpy(pred.astype(np.float32) / 2 ** 14).unsqueeze(0)
            pred_all.append(pred)
            titles_all.append(n.get_display_name())

        gt = process_fn.process(gt, meta_info)
        pred_all = [process_fn.process(p, meta_info) for p in pred_all]
        data = [{'images': [gt, ] + pred_all,
                 'titles': [burst_name, ] + titles_all}]
        cmd = vis.plot(data)

        if cmd == 'stop':
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the results on the denoising test sets. '
                                                 'setting_name denotes the name of the experiment setting, which '
                                                 'contains the list of methods for which to visualize results.')
    parser.add_argument('setting', type=str, help='Name of experiment setting')
    parser.add_argument('mode', type=str, help='grayscale or color')
    parser.add_argument('noise_level', type=str, help='Noise Level (can be 1, 2, 4, 8 or all)')

    args = parser.parse_args()

    print('Press \'n\' to show next image. \n'
          'Press \'q\' to quit. \n'
          'Zoom on a particular region by drawing a box around it (click on the two corner points). \n'
          'In the zoomed pane (last row), you can click on an image an drag around. \n'
          'Using \'w\' and \'s\' keys, you can navigate between the two panes (normal pane and zoom pane) \n'
          'Using the \'space\' key, you can toggle between showing all the images and showing only a single image. \n' 
          'In the single image mode, you can navigate between different images using the \'a\' and \'d\' keys. \n')

    visualize_results(args.setting, mode=args.mode, noise_level=int(args.noise_level))
