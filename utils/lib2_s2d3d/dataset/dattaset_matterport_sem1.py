import os
import glob
import numpy as np
from imageio import imread
# from shapely.geometry import LineString

import torch
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd

# from lib.misc import panostretch

__FOLD__ = {
    '1_train': [ "17DRP5sb8fy", "1LXtFkjw3qL", "1pXnuDYAj8r" ,"29hnd4uzFmX", "5LpN3gDmAk7",
                 "5q7pvUzZiYa", "759xd9YjKW5", "7y3sRwLe3Va", "82sE5b5pLXE", "8WUmhLawc2A",
                 "aayBHfsNo7d", "ac26ZMwG7aT", "B6ByNegPMKs", "b8cTxDM8gDG", "cV4RVeZvu5T",
                 "D7N2EKCX4Sj", "e9zR4mvMWw7", "EDJbREhghzL", "GdvgFV5R1Z5", "gTV8FGcVJC9",
                 "HxpKQynjfin", "i5noydFURQK", "JeFG25nYj2p", "JF19kD82Mey", "jh4fc5c5qoQ",
                 "kEZ7cmS4wCh", "mJXqzFtmKg4", "p5wJjkQkbXX", "Pm6F8kyY3z2", "pRbA3pwrgk9",
                 "PuKPg4mmafe", "PX4nDJXEHrG", "qoiz87JEwZ2", "rPc6DW4iMge", "s8pcmisQ38h",
                 "S9hNv5qa7GM", "sKLMLpTHeUy", "SN83YJsR3w2", "sT4fr6TAbpF", "ULsKaCPVFJR",
                 "uNb9QFRL6hY", "Uxmj2M2itWa", "V2XKFyX4ASd", "VFuaQ6m2Qom", "VVfe2KiqLaN",
                 "Vvot9Ly1tCj", "vyrNrziPKCB", "VzqfbhrpDEA", "XcA2TqTSSAj", "2n8kARJN3HM",
                 "D7G3Y4RVNrH", "dhjEzFoUFzH", "E9uDoFAP3SH", "gZ6f7yhEvPG", "JmbYfDe2QKZ",
                 "r1Q1Z4BcV1o", "r47D5H71a5s", "ur6pFq6Qu1A", "VLzqgDo317F", "YmJkqBEsHnH",
                 "ZMojNkEp431"],

    '1_valid': ['2azQ1b91cZZ', '8194nk5LbLH', 'EU6Fwq7SyZv', 'oLBMNvg9in8', 'QUCTc6BB5sX', 'TbHJrupSAjP', 'X7HyMhZNoso'],

    '1_test' : ['2t7WUuJeko7', '5ZKStnWn8Zo', 'ARNzJeq3xxb', 'fzynW3qQPVF', 'jtcxE69GiFV',
                'pa4otMbVnkk', 'q9vSo1VnCiC', 'rqfALeAoiTq', 'UwV83HsGsw3', 'wc2JMjhGNzB',
                'WYY7iVyf5p8', 'YFuZgdQ5vWj', 'yqstnuAEVhm', 'YVUC4YcDtcY', 'gxdoqLR6rwA',
                'gYvKGZ5eRqb', 'RPmz2sHmrrY', 'Vt2qJdWjCF2']
}



class matterport_SemDataset(data.Dataset):
    NUM_CLASSES = 41
    #ID2CLASS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
    ID2CLASS = ["wall", "floor", "chair", "door", "table", "picture", "cabinet", "cushion", "window", "sofa", "bed", "curtain", "chest_of_drawers", "plant", "sink", "stairs", "ceiling", "toilet", "stool", "towel", "mirror", "tv_monitor", "shower", "column", "bathtub", "counter", "fireplace", "lighting", "beam", "railing", "shelving", "blinds", "gym_equipment", "seating", "board_panel", "furniture", "appliances", "clothes", "objects", "misc", "unlabeled"]

    def __init__(self, root, fold, depth=False, hw=(512, 1024), mask_black=True, flip=False, rotate=False):
        assert fold in __FOLD__, 'Unknown fold'
        self.depth = depth
        self.hw = hw
        self.mask_black = mask_black
        self.rgb_paths = []
        self.sem_paths = []
        self.dep_paths = []

        for dname in __FOLD__[fold]:
            # print("path_root:",root,dname, glob.glob(os.path.join(root, dname, 'rgb', '*png')) , sorted(glob.glob(os.path.join(root, dname, 'rgb', '*png'))))
            self.rgb_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'rgb', '*png'))))
            self.sem_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'semantic', '*png'))))
            #self.dep_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'depth', '*png'))))
        #print("haha:",self.dep_paths)
        assert len(self.rgb_paths)
        #print("**:",len(self.rgb_paths), len(self.sem_paths), len(self.dep_paths))
        assert len(self.rgb_paths) == len(self.sem_paths)
        #assert len(self.rgb_paths) == len(self.dep_paths)
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

        # Convert all data to tensor
        out_dict = {
            'x': rgb,
            'sem': sem,
            'fname': os.path.split(self.rgb_paths[idx])[1].ljust(200),
        }
        return out_dict

class matterport_SemDataset22(data.Dataset):
    NUM_CLASSES = 41
    #ID2CLASS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
    ID2CLASS = ["wall", "floor", "chair", "door", "table", "picture", "cabinet", "cushion", "window", "sofa", "bed", "curtain", "chest_of_drawers", "plant", "sink", "stairs", "ceiling", "toilet", "stool", "towel", "mirror", "tv_monitor", "shower", "column", "bathtub", "counter", "fireplace", "lighting", "beam", "railing", "shelving", "blinds", "gym_equipment", "seating", "board_panel", "furniture", "appliances", "clothes", "objects", "misc", "unlabeled"]

    def __init__(self, root, fold, depth=True, hw=(512, 1024), mask_black=True, flip=False, rotate=False):
        assert fold in __FOLD__, 'Unknown fold'
        self.depth = depth
        self.hw = hw
        self.mask_black = mask_black
        self.rgb_paths = []
        self.sem_paths = []
        self.dep_paths = []
        for dname in __FOLD__[fold]:
            print("path_root:",root,dname, glob.glob(os.path.join(root, dname, 'rgb', '*png')) , sorted(glob.glob(os.path.join(root, dname, 'rgb', '*png'))))
            self.rgb_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'rgb', '*png'))))
            self.sem_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'semantic', '*png'))))
            self.dep_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'depth', '*png'))))
        print("haha:",self.dep_paths)
        assert len(self.rgb_paths)
        #print("**:",len(self.rgb_paths), len(self.sem_paths), len(self.dep_paths))
        assert len(self.rgb_paths) == len(self.sem_paths)
        assert len(self.rgb_paths) == len(self.dep_paths)
        self.flip = flip
        self.rotate = rotate

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = torch.FloatTensor(imread(self.rgb_paths[idx]) / 255.).permute(2,0,1)
        sem = torch.LongTensor(imread(self.sem_paths[idx])) - 1
        # sem_original = torch.LongTensor(imread(self.sem_paths[idx]))
        ### mapping ###################################################################################################

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

        # Convert all data to tensor
        out_dict = {
            'x': rgb,
            'sem': sem,
            'fname': os.path.split(self.rgb_paths[idx])[1].ljust(200),
        }
        return out_dict


class matterport_SemDataset33(data.Dataset):
    NUM_CLASSES = 20
    # ID2CLASS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
    ID2CLASS = ['wall', 'floor', 'chair', 'door', 'table', 'picture', 'furniture', 'objects', 'window', 'sofa', 'bed', 'sink', 'stairs', 'ceiling', 'toilet', 'mirror', 'shower', 'bathtub', 'counter', 'shelving']


    def __init__(self, root, fold, depth=True, hw=(512, 1024), mask_black=True, flip=False, rotate=False):
        assert fold in __FOLD__, 'Unknown fold'
        self.depth = depth
        self.hw = hw
        self.mask_black = mask_black
        self.rgb_paths = []
        self.sem_paths = []
        self.dep_paths = []
        for dname in __FOLD__[fold]:
            print("path_root:", root, dname, glob.glob(os.path.join(root, dname, 'rgb', '*png')),
                  sorted(glob.glob(os.path.join(root, dname, 'rgb', '*png'))))
            self.rgb_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'rgb', '*png'))))
            self.sem_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'semantic', '*png'))))
            self.dep_paths.extend(sorted(glob.glob(os.path.join(root, dname, 'depth', '*png'))))
        print("haha:", self.dep_paths)
        assert len(self.rgb_paths)
        # print("**:",len(self.rgb_paths), len(self.sem_paths), len(self.dep_paths))
        assert len(self.rgb_paths) == len(self.sem_paths)
        assert len(self.rgb_paths) == len(self.dep_paths)
        self.flip = flip
        self.rotate = rotate

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb = torch.FloatTensor(imread(self.rgb_paths[idx]) / 255.).permute(2, 0, 1)
        sem = torch.LongTensor(imread(self.sem_paths[idx])) - 1
        # sem_original = torch.LongTensor(imread(self.sem_paths[idx]))

        ### mapping ####################################################################################################
        print("sem:", sem.shape, np.unique(sem))
        df = pd.read_csv("eigen13_mapping_from_mpcat40.csv")
        print("df:", df.head())
        sem_array = np.asarray(sem)

        for j in range(42):
            labels = j
            itemindex = np.where((sem_array == j))
            # print("itemindex:", itemindex[0], itemindex[1])
            row = itemindex[0]
            column = itemindex[1]
            new_label = df.loc[(df["mpcat40index"] == j, ["eigen13id"])]
            new_label = np.array(new_label)[0][0].astype(np.uint8)

            print("new_labels:", new_label, type(new_label))
            sem_array[row, column] = new_label

            # print("sem_original[row]:", np.unique(sem), sem.shape)

        labels_of_all = sem_array - 100
        ################################################################################################################
        sem = torch.from_numpy(labels_of_all)
        print("sem_:", np.unique(sem))

        if self.depth:
            dep = imread(self.dep_paths[idx])
            dep = np.where(dep == 65535, 0, dep / 512)
            dep = np.clip(dep, 0, 4)
            dep = torch.FloatTensor(dep[None])
            rgb = torch.cat([rgb, dep], 0)
        H, W = rgb.shape[1:]

        if (H, W) != self.hw:
            rgb = F.interpolate(rgb[None], size=self.hw, mode='bilinear', align_corners=False)[0]
            sem = F.interpolate(sem[None, None].float(), size=self.hw, mode='nearest')[0, 0].long()

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

        # Convert all data to tensor
        out_dict = {
            'x': rgb,
            'sem': sem,
            'fname': os.path.split(self.rgb_paths[idx])[1].ljust(200),
        }
        return out_dict
