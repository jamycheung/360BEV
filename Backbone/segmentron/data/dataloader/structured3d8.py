"""Structured3D synthetic panoramic Dataset."""
import os
import logging
import torch
import numpy as np
import glob
import json
from PIL import Image
from segmentron.data.dataloader.seg_data_base import SegmentationDataset

class Structured3d8Segmentation(SegmentationDataset):
    """Structured3d Semantic Segmentation Dataset."""
    NUM_CLASS = 8

    def __init__(self, root='datasets/Structured3D/Structured3D', split='train', mode=None, transform=None, **kwargs):
        super(Structured3d8Segmentation, self).__init__(root, split, mode, transform, **kwargs)
        root = self.root
        assert os.path.exists(root), "Please put the data in {SEG_ROOT}"
        self.images, self.masks = _get_structured3d_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in {}".format(os.path.join(root, split)))
        logging.info('Found {} images in {}'.format(len(self.images), os.path.join(root, split)))

        self._key = np.array([-1,6,3,-1,-1,1,4,5,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

    def _map40to8(self, mask):
        values = np.unique(mask)
        for value in values:
            if value == 255: 
                mask[mask==value] = -1
            else:
                mask[mask==value] = self._key[value]
        return mask

    def _val_sync_transform_resize(self, img, mask):
        # short_size = self.crop_size
        # img = img.resize(short_size, Image.BICUBIC)
        # mask = mask.resize(short_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask, resize=True)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform_resize(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._val_sync_transform_resize(img, mask)
        if self.transform is not None:
            img = self.transform(img)

        mask[mask == 255] = -1 # ignore 255
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._map40to8(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ('ceiling', 'chair',
                'door', 'floor', 'sofa',
                'table', 'wall', 'window')

def _get_structured3d_pairs(folder, split='train'):
    '''image is jpg, label is png'''
    # datasets/Structured3D/Structured3D/scene_03400/2D_rendering/55705/panorama/full/rgb_rawlight.png
    # datasets/Structured3D/Structured3D/scene_03400/2D_rendering/55705/panorama/full/rgb_warmlight.png
    # datasets/Structured3D/Structured3D/scene_03400/2D_rendering/55705/panorama/full/rgb_coldlight.png
    # datasets/Structured3D/Structured3D/scene_03400/2D_rendering/55705/panorama/full/albedo.png
    # datasets/Structured3D/Structured3D/scene_03400/2D_rendering/55705/panorama/full/depth.png
    # datasets/Structured3D/Structured3D/scene_03400/2D_rendering/55705/panorama/full/semantic.png
    # datasets/Structured3D/Structured3D/scene_00200/2D_rendering/161/panorama/full/rgb_rawlight.png
    # img_paths = glob.glob(os.path.join(*[folder, '*', '2D_rendering', '*', 'panorama/full/rgb_rawlight.png']))
    # mask_paths = glob.glob(os.path.join(*[folder, '*', '2D_rendering', '*', 'panorama/full/semantic.png']))
    # img_paths = sorted(img_paths)
    # mask_paths = sorted(mask_paths)
    img_paths = []
    mask_paths = []
    gt_file_txt = os.path.join(folder, '{}.txt'.format(split))
    with open(gt_file_txt) as f:
        mask_paths = [line.rstrip() for line in f]
    mask_paths = [os.path.join(folder, m) for m in mask_paths]
    img_paths = [m.replace('semantic.png', 'rgb_rawlight.png') for m in mask_paths]
    return img_paths, mask_paths



if __name__ == '__main__':
    from torchvision import transforms
    import torch.utils.data as data
     # Transforms for Normalization
    input_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225)),])
     # Create Dataset
    trainset = Structured3d8Segmentation(split='train', transform=input_transform)
     # Create Training Loader
    train_data = data.DataLoader(trainset, 2, shuffle=True, num_workers=0)
    for i, data in enumerate(train_data):
        imgs, targets, _ = data
        print(imgs.shape)
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
