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
import torchvision.transforms as transforms

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])
file_folder_name = 'smnet_training_data_zteng'
# file_folder_gt_name = 'topdown_gt_train'
file_folder_gt_name = 'topdown_gt_real_height'

normalize = transforms.Compose([
    # transforms.ToPILImage(),
    # # Addblur(p=1, blur="Gaussian"),
    # AddSaltPepperNoise(0.05, 1),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

class DatasetLoader_pano(data.Dataset):
    def __init__(self, cfg, split='train', resize_index = False):
        self.split = split
        self.resize_index = resize_index

        if split == 'train':
            self.root = cfg['root'] + '/training'
            self.resize_index = True
        elif split == 'val':
            self.root = cfg['root'] + '/valid'
        elif split == 'test':
            self.root = cfg['root'] + '/testing'

        # self.ego_downsample = cfg['ego_downsample']
        self.feature_type = cfg['feature_type']
        self.files = os.listdir(os.path.join(self.root, file_folder_name))
        self.df = pd.read_csv("eigen13_mapping_from_mpcat40.csv")

        #self.files = np.array([x for x in self.files if '_'.join(x.split('_')[:2]) in envs_splits['{}_envs'.format(split)]])  # using numpy format
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

        h5file = h5py.File(os.path.join(self.root, file_folder_name, file), 'r')
        rgb = np.array(h5file['rgb'])
        depth = np.array(h5file['depth'])
        h5file.close()

        ################################################################################################################
        h5file = h5py.File(os.path.join(self.root, file_folder_gt_name, file), 'r')
        semmap_gt = np.array(h5file['map_semantic'])  # 40 classes 40 -> 20 classes

        for j in range(42):
            labels = j
            itemindex = np.where((semmap_gt == j))
            # print("itemindex:", itemindex[0], itemindex[1])
            row = itemindex[0]
            column = itemindex[1]
            new_label = self.df.loc[(self.df["mpcat40index"] == j, ["eigen13id"])]
            new_label = np.array(new_label)[0][0].astype(np.uint8)

            # print("new_labels:", new_label, type(new_label))
            semmap_gt[row, column] = new_label

            # print("sem_original[row]:", np.unique(sem), sem.shape)

        semmap_gt = semmap_gt - 100
        # print('semmap_gt:', semmap_gt.shape, np.unique(semmap_gt))

        # modified
        # h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_maxHIndices_{}'.format(self.split), file), 'r')
        h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_maxHIndices'.format(self.split), file), 'r')

        proj_indices = np.array(h5file['indices'])
        masks_outliers = np.array(h5file['masks_outliers'])
        h5file.close()

        rgb_no_norm = rgb

        rgb_img = rgb.astype(np.float32)
        rgb_img = rgb_img / 255.0


        # print('rgb_shape_in_dataloader:', rgb_img.shape)

        rgb_img = torch.FloatTensor(rgb_img).permute(2, 0, 1)
        rgb_img = normalize(rgb_img)

        # rgb_img = rgb_img.unsqueeze(0)

        depth_img = depth
        depth_img = depth_img.astype(np.float32)
        depth_img = torch.FloatTensor(depth_img).unsqueeze(0)
        depth_img = depth_normalize(depth_img)
        # depth_img = depth_img.unsqueeze(0)

        # Rgb_img = []
        # Depth_img = []

        # for i in range(0, 4):
        #     rgb_img = rgb[i]
        #     rgb_img = rgb_img.astype(np.float32)
        #     rgb_img = rgb_img / 255.0
        #
        #     rgb_img = torch.FloatTensor(rgb_img).permute(2, 0, 1)
        #     rgb_img = normalize(rgb_img)
        #     rgb_img = rgb_img.unsqueeze(0)
        #     Rgb_img.append(rgb_img)
        #
        #
        #     depth_img = depth[i]
        #     depth_img = depth_img.astype(np.float32)
        #     depth_img = torch.FloatTensor(depth_img).unsqueeze(0)
        #     depth_img = depth_normalize(depth_img)
        #     depth_img = depth_img.unsqueeze(0)
        #     Depth_img.append(depth_img)
        #
        # RGB = np.concatenate(Rgb_img, axis=0)
        # DEPTH = np.concatenate(Depth_img, axis=0)


        rgb = rgb_img


        # if self.resize_index ==  True:
            
        #     h, w = rgb.size(2), rgb.size(3)
        #     short_size = random.randint(int(h * 0.5), int(h * 2))

        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
    
        #     # rgb = rgb.resize((oh, ow), Image.BILINEAR)
        #     rgb = F.interpolate(rgb, size=(oh, ow), mode="bilinear", align_corners=True)
        #     # mask = mask.resize((ow, oh), Image.NEAREST)

        depth = depth_img
        #print("depth:", depth.size(), depth)

        proj_indices = torch.from_numpy(proj_indices).long()
        masks_outliers = torch.from_numpy(masks_outliers).bool()

        masks_inliers = ~masks_outliers
        ################ semmap_gt input ###############################################################################
        # semmap_index = self.semmap_GT_indx[env_index]
        # semmap = self.semmap_GT[semmap_index]
        # semmap = torch.from_numpy(semmap).long()
        return (rgb, rgb_no_norm, masks_inliers, proj_indices, semmap_gt)


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
        self.files_gt = os.listdir(os.path.join(self.root, file_folder_gt_name))
        # print('files_gt:', len(self.files_gt), len(self.files))

        assert len(self.files) == len(self.files_gt)
        assert len(self.files) > 0

        self.available_idx = np.array(list(range(len(self.files))))

    def __len__(self):
        return len(self.available_idx)

    def __getitem__(self, index):
        env_index = self.available_idx[index]

        file = self.files[env_index]
        env = self.envs[env_index]

        h5file = h5py.File(os.path.join(self.root, file_folder_name, file), 'r')
        rgb = np.array(h5file['rgb'])


        depth = np.array(h5file['depth'])
        h5file.close()

        ################################################################################################################
        h5file = h5py.File(os.path.join(self.root, file_folder_gt_name, file), 'r')
        # print('h5file_gt:', self.root, file_folder_gt_name)

        semmap_gt = np.array(h5file['map_semantic'])  # 40 classes 40 -> 20 classes
        map_mask = np.array(h5file['mask'])
        map_heights = np.array(h5file['map_heights'])
        h5file.close()


        for j in range(42):
            labels = j
            itemindex = np.where((semmap_gt == j))
            # print("itemindex:", itemindex[0], itemindex[1])
            row = itemindex[0]
            column = itemindex[1]
            new_label = self.df.loc[(self.df["mpcat40index"] == j, ["eigen13id"])]
            new_label = np.array(new_label)[0][0].astype(np.uint8)

            # print("new_labels:", new_label, type(new_label))
            semmap_gt[row, column] = new_label

        semmap_gt = semmap_gt - 100
        # print('semmap_gt:', semmap_gt.shape, np.unique(semmap_gt))

        # modified
        # h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_maxHIndices_{}'.format(self.split), file), 'r')
        h5file = h5py.File(os.path.join(self.root, 'smnet_training_data_maxHIndices'.format(self.split), file), 'r')

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

        depth_img = depth
        depth_img = depth_img.astype(np.float32)
        depth_img = torch.FloatTensor(depth_img).unsqueeze(0)
        depth_img = depth_normalize(depth_img)
        depth_img = depth_img.unsqueeze(0)


        rgb = rgb_img
        #print('rgb:', rgb.size(), rgb)
        depth = depth_img

        proj_indices = torch.from_numpy(proj_indices).long()
        masks_outliers = torch.from_numpy(masks_outliers).bool()

        masks_inliers = ~masks_outliers

        ################ semmap_gt input ###############################################################################
        # semmap_index = self.semmap_GT_indx[env_index]
        # semmap = self.semmap_GT[semmap_index]
        # semmap = torch.from_numpy(semmap).long()

        return (rgb, rgb_no_norm, masks_inliers, proj_indices, semmap_gt,  map_mask, map_heights)




