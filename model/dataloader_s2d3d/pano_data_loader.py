import os
import h5py
import torch
import numpy as np
from torch.utils import data
import cv2
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

##################################################################################################

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])
file_folder_gt_name = 'bev'

normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


class DatasetLoader_pano_detr(data.Dataset):

    def __init__(self, cfg, split='train'):
        self.split = split
        self.root = cfg['root']

        if split == 'train':
            self.split_dir = cfg['root'] + '/train'

        elif split == 'val':
            self.split_dir = cfg['root'] + '/val'
        elif split == 'test':
            self.split_dir = cfg['root'] + '/test'

        self.feature_type = cfg['feature_type']

        self.files = os.listdir(self.split_dir)

        self.files = np.array(self.files)
        self.envs = np.array([x.split('.')[0]
                              for x in self.files])  # using numpy format

        self.files_gt = os.listdir(os.path.join(self.root,
                                                file_folder_gt_name))

        self.available_idx = np.array(list(range(len(self.files))))

    def __len__(self):
        return len(self.available_idx)

    def __getitem__(self, index):
        env_index = self.available_idx[index]

        file = self.files[env_index]
        env = self.envs[env_index]

        for i in self.files_gt:
            if env in i:
                gt_file_name = i

        #rgb
        rgb = cv2.imread(os.path.join(self.split_dir, file))

        #bev
        h5file = h5py.File(
            os.path.join(self.root, file_folder_gt_name, gt_file_name), 'r')

        rotationz = np.array(h5file['rotation_z'])
        rotationz = rotationz[0]
        camera_location = np.array(h5file['camera_location'])
        camera_height_z = camera_location[2]

        semmap_gt = np.array(h5file['labels'])
        map_mask = np.array(h5file['mask'])
        proj_indices = np.array(h5file['indices'])
        map_heights = np.array(h5file['map_heights'])
        map_heights = map_heights - camera_height_z

        ############################### Rotate the mask and Height in the dataloader ###################################
        degree = 180 - rotationz * 180 / torch.pi
        degree = -degree.item()

        semmap_gt = torchvision.transforms.ToPILImage()(semmap_gt)
        semmap_gt = torchvision.transforms.functional.rotate(semmap_gt,
                                                             angle=degree,
                                                             expand=False,
                                                             center=None,
                                                             fill=0)

        semmap_gt = torchvision.transforms.ToTensor()(semmap_gt)
        semmap_gt = semmap_gt.squeeze(0)
        semmap_gt = semmap_gt.long()

        map_mask = map_mask.astype(np.int32)
        map_mask = torchvision.transforms.ToPILImage()(map_mask)

        map_mask = torchvision.transforms.functional.rotate(map_mask,
                                                            angle=degree,
                                                            expand=False,
                                                            center=None,
                                                            fill=0)
        map_mask = torchvision.transforms.ToTensor()(map_mask)
        map_mask = map_mask.squeeze(0)

        map_heights = torchvision.transforms.ToPILImage()(map_heights)
        map_heights = torchvision.transforms.functional.rotate(map_heights,
                                                               angle=degree,
                                                               expand=False,
                                                               center=None,
                                                               fill=0)

        map_heights = torchvision.transforms.ToTensor()(map_heights)
        map_heights = map_heights.squeeze(0)

        h5file.close()

        rgb_no_norm = rgb
        rgb_img = rgb.astype(np.float32)
        rgb_img = rgb_img / 255.0

        rgb_img = torch.FloatTensor(rgb_img).permute(2, 0, 1)
        rgb_img = normalize(rgb_img)

        rgb = rgb_img

        proj_indices = torch.from_numpy(proj_indices).long()

        return (rgb, rgb_no_norm, proj_indices, semmap_gt, map_mask,
                map_heights)
