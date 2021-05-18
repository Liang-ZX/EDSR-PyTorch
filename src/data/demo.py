import os

from data import common

import numpy as np
import imageio

import torch
import torch.utils.data as data

class Demo(data.Dataset):
    def __init__(self, args, name='Demo', train=False, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = False
        self.benchmark = benchmark

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.find('.png') >= 0 or f.find('.jp') >= 0:
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename, suffix = os.path.splitext(os.path.basename(self.filelist[idx]))
        lr = imageio.imread(self.filelist[idx])
        lr = get_patch(lr)

        if not self.args.test_pair:
            lr, = common.set_channel(lr, n_channels=self.args.n_colors)
            lr_t, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
            return lr_t, -1, filename
        else:
            hr_filename = filename[:-2] + suffix
            hr_filepath = os.path.join(os.path.dirname(self.filelist[idx]), "HR", hr_filename)
            hr = imageio.imread(hr_filepath)
            hr = get_patch(hr, isHR=True)
            if self.args.model == "Bicubic":
                imageio.imwrite("~/code/LightWeightSR/EDSR-PyTorch/experiment/test/results-Demo/223061_local.png", hr)
            pair = (lr, hr)
            pair = common.set_channel(*pair, n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            return pair_t[0], pair_t[1], filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale


def get_patch(img, isHR=False):
    leftup_pos = [32, 27]
    rightdown_pos = [97, 67]
    if isHR:
        leftup_pos = [i*4 for i in leftup_pos]
        rightdown_pos = [i*4 for i in rightdown_pos]
    return img[leftup_pos[1]:rightdown_pos[1], leftup_pos[0]:rightdown_pos[0], :]
