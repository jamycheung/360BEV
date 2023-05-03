import os
import glob
import numpy as np
from imageio import imread

import torch
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
import cv2


__FOLD__ = {
    '1_train': ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'],
    '1_valid': ['area_5a', 'area_5b'],
    '2_train': ['area_1', 'area_3', 'area_5a', 'area_5b', 'area_6'],
    '2_valid': ['area_2', 'area_4'],
    '3_train': ['area_2', 'area_4', 'area_5a', 'area_5b'],
    '3_valid': ['area_1', 'area_3', 'area_6'],
}

class S2d3dSemDataset(data.Dataset):
    NUM_CLASSES = 13
    ID2CLASS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
    def __init__(self, cfg_dict, Split,  depth=False, hw=(512, 1024), mask_black=True, flip=False, rotate=False):
        
        root = cfg_dict["root"]

        split = Split.split('_')
        fold_index = split[0]
        fold_type = split[1]

        if fold_type == 'train':
            self.flip = True
            self.rotate = True
            if fold_index == '1':
                fold = '1_train'
            elif fold_index == '2':
                fold = '2_train'
            elif fold_index == '3':
                fold = '3_train'
        elif fold_type == 'val':
            if fold_index == '1':
                fold = '1_valid'
            elif fold_index == '2':
                fold = '2_valid'
            elif fold_index == '3':
                fold = '3_valid'

        assert fold in __FOLD__, 'Unknown fold'
        self.depth = depth
        self.hw = hw
        self.mask_black = mask_black
        self.rgb_paths = []
        self.sem_paths = []
        #self.dep_paths = []
        for dname in __FOLD__[fold]:
            self.rgb_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'rgb', '*png'))))
            self.sem_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'semantic', '*png'))))
            # self.dep_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'depth', '*png'))))
        assert len(self.rgb_paths)
        assert len(self.rgb_paths) == len(self.sem_paths)
        # assert len(self.rgb_paths) == len(self.dep_paths)
        self.flip = flip
        self.rotate = rotate

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = torch.FloatTensor(imread(self.rgb_paths[idx]) / 255.).permute(2,0,1)
        sem = torch.LongTensor(imread(self.sem_paths[idx])) - 1
        if self.depth:
            dep = imread(self.dep_paths[idx])
            dep = np.where(dep==65535, 0, dep/512)
            dep = np.clip(dep, 0, 4)
            dep = torch.FloatTensor(dep[None])
            rgb = torch.cat([rgb, dep], 0)
        H, W = rgb.shape[1:]
        if (H, W) != self.hw:
            rgb = F.interpolate(rgb[None], size=self.hw, mode='bilinear', align_corners=False)[0]
            sem = F.interpolate(sem[None,None].float(), size=self.hw, mode='nearest')[0,0].long()

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            rgb = torch.flip(rgb, (-1,))
            sem = torch.flip(sem, (-1,))

        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(W)
            rgb = torch.roll(rgb, dx, dims=-1)
            sem = torch.roll(sem, dx, dims=-1)

        # Mask out top-down black
        if self.mask_black:
            sem[rgb.sum(0) == 0] = -1

        fname = os.path.split(self.rgb_paths[idx])[1].ljust(200)

        # Convert all data to tensor
        out_dict = {
            'x': rgb,
            'sem': sem,
            'fname': os.path.split(self.rgb_paths[idx])[1].ljust(200),
        }
        # return out_dict
        return rgb, sem, fname
