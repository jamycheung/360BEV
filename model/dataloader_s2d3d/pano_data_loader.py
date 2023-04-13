import os
import h5py
import json
import torch
import numpy as np
import torch.nn.functional as F
import random
from PIL import Image, ImageFilter
from torch.utils import data
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# envs_splits = json.load(open('data/envs_splits.json', 'r'))
##################################################################################################

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])
file_folder_name = 'data_base_with_rotationz_realdepth'
file_folder_gt_name = 'ground_truth'

normalize = transforms.Compose([
    # transforms.ToPILImage(),
    # # Addblur(p=1, blur="Gaussian"),
    # AddSaltPepperNoise(0.05, 1),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

class DatasetLoader_pano_detr(data.Dataset):
    def __init__(self, cfg, split='train'):
        self.split = split

        if split == 'train':
            self.root = cfg['root'] + '/training'

        elif split == 'val':
            self.root = cfg['root'] + '/valid'
        elif split == 'test':
            self.root = cfg['root'] + '/testing'

        # self.ego_downsample = cfg['ego_downsample']
        self.feature_type = cfg['feature_type']

        self.files = os.listdir(os.path.join(self.root, file_folder_name))
        self.df = pd.read_csv("eigen13_mapping_from_mpcat40.csv")


        self.files = np.array(self.files)
        self.envs = np.array([x.split('.')[0] for x in self.files])  # using numpy format
        # print('files_files:', self.files)

        # -- load semantic map GT
        # h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_semmap.h5'), 'r')
        # self.semmap_GT = np.array(h5file['semantic_maps'])
        # h5file.close()
        # self.semmap_GT_envs = json.load(open(os.path.join(self.root, 'smnet_training_data_semmap.json'), 'r'))
        # self.semmap_GT_indx = {i: self.semmap_GT_envs.index(self.envs[i] + '.h5') for i in range(len(self.files))}
        self.files_gt = os.listdir(os.path.join(self.root, file_folder_gt_name))

        assert len(self.files) == len(self.files_gt)
        assert len(self.files) > 0

        self.available_idx = np.array(list(range(len(self.files))))

    def __len__(self):
        return len(self.available_idx)

    def __getitem__(self, index):
        env_index = self.available_idx[index]

        file = self.files[env_index]
        env = self.envs[env_index]
        env_index = env.split('_')[1]
        # print('env_index:', env_index)


        for i in self.files_gt:
            if env_index in i:
                gt_file_name = i


        h5file = h5py.File(os.path.join(self.root, file_folder_name, file), 'r')
        rgb = np.array(h5file['rgb'])

        rotationz = np.array(h5file['rotation_z'])
        rotationz = rotationz[0]

        camera_location = np.array(h5file['camera_location'])
        camera_height_z = camera_location[2]
        # print('camera_height:', camera_height_z)

        # depth = np.array(h5file['depth'])
        h5file.close()

        ################################################################################################################
        h5file = h5py.File(os.path.join(self.root, file_folder_gt_name, gt_file_name), 'r')
        # print('h5file_gt:', self.root, file_folder_gt_name)

        semmap_gt = np.array(h5file['map_semantic'])  # 40 classes 40 -> 20 classes
        map_mask = np.array(h5file['mask'])
        map_heights = np.array(h5file['map_heights'])
        map_heights = map_heights - camera_height_z


        ############# 在dataloader里面旋转mask，和Height ###################################
        degree = 180 - rotationz * 180 / torch.pi
        degree = - degree.item()

        # semmap_gt = semmap_gt.short()
        # print('semmap_gt_0:', np.unique(semmap_gt), semmap_gt.shape, semmap_gt.dtype)

        semmap_gt = torchvision.transforms.ToPILImage()(semmap_gt)
        semmap_gt = torchvision.transforms.functional.rotate(semmap_gt, angle=degree, expand=False, center=None, fill=0)

        semmap_gt = torchvision.transforms.ToTensor()(semmap_gt)
        semmap_gt = semmap_gt.squeeze(0)
        semmap_gt = semmap_gt.long()

        ######################################## *****************************************
        # print('map_mask_-1:', map_mask.dtype, map_mask.shape, semmap_gt.shape)

        map_mask = map_mask.astype(np.int32)
        map_mask = torchvision.transforms.ToPILImage()(map_mask)

        map_mask = torchvision.transforms.functional.rotate(map_mask, angle=degree, expand=False, center=None, fill=0)
        map_mask = torchvision.transforms.ToTensor()(map_mask)
        # print('map_mask_0:', torch.unique(map_mask), map_mask.size())
        map_mask = map_mask.squeeze(0)

        ######################################## *****************************************
        map_heights = torchvision.transforms.ToPILImage()(map_heights)
        map_heights = torchvision.transforms.functional.rotate(map_heights, angle=degree, expand=False, center=None, fill=0)

        map_heights = torchvision.transforms.ToTensor()(map_heights)
        map_heights = map_heights.squeeze(0)

        # print('map_heights:', map_heights.shape, torch.unique(map_heights))

        h5file.close()


        # modified
        # h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_maxHIndices_{}'.format(self.split), file), 'r')
        h5file = h5py.File(os.path.join(self.root, 'Indices_realdepth'.format(self.split), file), 'r')

        proj_indices = np.array(h5file['indices'])
        masks_outliers = np.array(h5file['masks_outliers'])
        h5file.close()

        rgb_no_norm = rgb

        rgb_img = rgb.astype(np.float32)
        rgb_img = rgb_img / 255.0
        # print('rgb_shape_in_dataloader:', rgb_img.shape, rgb_img)

        rgb_img = torch.FloatTensor(rgb_img).permute(2, 0, 1)
        rgb_img = normalize(rgb_img)


        # rgb_img = rgb_img.unsqueeze(0)

        # depth_img = depth
        # depth_img = depth_img.astype(np.float32)
        # depth_img = torch.FloatTensor(depth_img).unsqueeze(0)
        # depth_img = depth_normalize(depth_img)
        # depth_img = depth_img.unsqueeze(0)


        rgb = rgb_img
        # print('rgb:', rgb.size(), rgb)
        # depth = depth_img
        # print("depth:", depth.size(), depth)


        proj_indices = torch.from_numpy(proj_indices).long()
        masks_outliers = torch.from_numpy(masks_outliers).bool()
        masks_inliers = ~masks_outliers

        ################ semmap_gt input ###############################################################################
        # semmap_index = self.semmap_GT_indx[env_index]
        # semmap = self.semmap_GT[semmap_index]
        # semmap = torch.from_numpy(semmap).long()

        return (rgb, rgb_no_norm, masks_inliers, proj_indices, semmap_gt, rotationz, map_mask, map_heights, env_index)




