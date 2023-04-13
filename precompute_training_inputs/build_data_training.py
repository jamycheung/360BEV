import os
import sys
import h5py
import torch
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from utils.projector.pcd_generator_from_depth import Point_Saver

# from utils.habitat_utils import HabitatUtils
from utils.lib2_mp3d.config import config, update_config
from utils.lib2_mp3d import dataset

import torchvision.transforms as transforms


# -- settings
# output_dir = 'data/training/smnet_training_data_zteng/'
# output_dir = "/cvhci/data/VisLoc/zteng/trans4map_baseline/training/smnet_training_data_zteng"

# output_dir = "/cvhci/data/VisLoc/zteng/trans4map_baseline/testing/smnet_training_data_zteng"
output_dir = "/cvhci/data/VisLoc/zteng/trans4map_baseline/testing/smnet_training_data_zteng"

os.makedirs(output_dir, exist_ok=True)

########################################################################################################################
# Parse args & config
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--cfg', required=True)
# parser.add_argument('--pth')
# arser.add_argument('--out')
# parser.add_argument('--vis_dir')

# parser.add_argument('--y', action='store_true')
# parser.add_argument('--test_hw', type=int, nargs='*')

parser.add_argument('opts',
                    help='Modify config options using the command-line',
                    default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
update_config(config, args)

########################################################################################################################
#Settings

resolution = 0.02 # topdown resolution
z_clip = 0.70 # detections over z_clip will be ignored
features_spatial_dimensions = (1024, 2048)

# -- Create model
normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])


# -- build projector
map_world_shift = np.zeros(3)
# world_shift_origin=torch.from_numpy(map_world_shift).float().to(device=device)


projector = Point_Saver(features_spatial_dimensions[0], features_spatial_dimensions[1] ,z_clip)

##### 返回一个 pcd, 一个no_depth_mask


################################################# inital Dataset #######################################################
# Init dataset
DatasetClass = getattr(dataset, config.dataset.name)
config.dataset.train_kwargs.update(config.dataset.common_kwargs)

print("config.dataset.train_kwargs:", config.dataset.train_kwargs)

train_dataset = DatasetClass(**config.dataset.train_kwargs)

train_loader = DataLoader(train_dataset, 1,
                          num_workers=config.num_workers,
                          pin_memory=config.cuda)
# device = torch.device('cuda')
device = torch.device('cpu')

"""
 -->> START
"""

for batch in tqdm(train_loader, position=1, total=len(train_loader)):

    camera_location = [0, 0, 0]

    color_dep = batch['x'].to(device)
    # print('color_dep:', color_dep[:,:,:10,:10])

    name_name = batch['fname'][0]
    # print('name_name:', name_name)
    # sem = batch['sem'].to(device)

    rgb =  color_dep[:,:3,:,:].float()
    # print('shape_rgb_depth1:', rgb.size(), rgb)
    # rgb = normalize(rgb)
    rgb = rgb.permute(0,2,3,1)
    rgb = rgb.squeeze().to(device)


    dep = color_dep[:,3,:,:].float()
    ## dep =  depth_normalize(dep)
    # print('depth2:', dep.size(), dep[:, 500:600, 1200:1400])
    dep = dep.squeeze().to(device)

    ####################################################################################################################

    ## 通过实例化的projector 得到点云！
    pc, mask, XYZ = projector.forward(dep , camera_location, name_name)

    # pc = pc.cpu().numpy()
    mask_outliers = mask

    projection_indices = pc
    # print('projection_indices:', projection_indices)

    masks_outliers = mask_outliers

    file_index_index = name_name.split('.')[0]
    # print('file_index_index:', file_index_index, type(name_name))

    filename = os.path.join(output_dir, file_index_index + '.h5')
    with h5py.File(filename, 'w') as f:
        f.create_dataset('rgb', data=rgb, dtype=np.uint8)
        f.create_dataset('depth', data=dep, dtype=np.float32)
        f.create_dataset('XYZ', data = XYZ, dtype=np.float32)

        f.create_dataset('projection_indices', data=pc, dtype=np.float32)
        f.create_dataset('masks_outliers', data=masks_outliers, dtype=np.bool)





