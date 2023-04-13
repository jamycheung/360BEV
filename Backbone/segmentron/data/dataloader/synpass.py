"""Prepare SynPASS dataset"""
import os
import torch
import numpy as np
import logging

import torchvision
from PIL import Image
from segmentron.data.dataloader.seg_data_base import SegmentationDataset
import random
from torch.utils import data
import glob

class SynPASSSegmentation(SegmentationDataset):
    """SynPASS Semantic Segmentation Dataset."""
    NUM_CLASS = 22

    def __init__(self, root='datasets/SynPASS', split='val', mode=None, transform=None, weather='all', **kwargs):
        super(SynPASSSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/SynPASS"
        self.root = root
        self.weather = weather
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split, self.weather)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self._key = np.array([-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])

    def _map23to22(self, mask):
        values = np.unique(mask)
        new_mask = np.zeros_like(mask)
        new_mask -= 1
        for value in values:
            if value == 255: 
                new_mask[mask==value] = -1
            else:
                new_mask[mask==value] = self._key[value]
        mask = new_mask
        return mask
    def _val_sync_transform_resize(self, img, mask):
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)
        if self.transform is not None:
            img = self.transform(img)

        return img, mask, os.path.basename(self.images[index])
    
    def _mask_transform(self, mask):
        target = self._map23to22(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('Building','Fence','Other','Pedestrian','Pole','RoadLine',
        'Road','SideWalk','Vegetation','Vehicles',
        'Wall','TrafficSign','Sky','Ground','Bridge','RailTrack',
        'GroundRail','TrafficLight','Static','Dynamic','Water','Terrain',)


def _get_city_pairs(folder, split='train', weather='all'):
    # datasets/SynPASS/img/cloud/train/MAP_1_point2/000000.jpg
    img_paths = glob.glob(os.path.join(*[folder, 'img', '*', split, '*', '*.jpg']))
    # datasets/SynPASS/semantic/cloud/train/MAP_1_point2/000000_trainID.png
    mask_paths = glob.glob(os.path.join(*[folder, 'semantic', '*', split, '*', '*_trainID.png']))
    assert len(img_paths)==len(mask_paths)
    if weather in ['/cloud', '/fog', '/rain', '/sun']:
        img_paths = [m for m in img_paths if weather in m]
        mask_paths = [m for m in mask_paths if weather in m]
    if weather in ['day', 'night']:
        new_img_paths = []
        new_mask_paths = []
        all_map_dn_val = read_list(os.path.join(folder, 'all_map_{}_val.txt'.format(weather)))
        for img, mask in zip(img_paths, mask_paths):
            _, _, _, _, _, p, _ = img.split('/')
            if p in all_map_dn_val:
                new_img_paths.append(img)
                new_mask_paths.append(mask)
        img_paths = new_img_paths
        mask_paths = new_mask_paths
    img_paths = sorted(img_paths)    
    mask_paths = sorted(mask_paths)
    assert len(img_paths) == len(mask_paths)
    logging.info('Found {} images in the folder {}'.format(len(img_paths), folder))
    return img_paths, mask_paths

def read_list(t):
    with open(t) as f:
        l = [line.rstrip() for line in f]
    return l

if __name__ == '__main__':
    dst = SynPASSSegmentation(split='train', mode='train')
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels, *args = data
        break