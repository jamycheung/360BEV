"""Stanford2D3D Pinhole Dataset."""
import os
import logging
import torch
import numpy as np
import glob
import json
from PIL import Image
from segmentron.data.dataloader.seg_data_base import SegmentationDataset

__FOLD__ = {
    '1_train': ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'],
    '1_val': ['area_5a', 'area_5b'],
    '2_train': ['area_1', 'area_3', 'area_5a', 'area_5b', 'area_6'],
    '2_val': ['area_2', 'area_4'],
    '3_train': ['area_2', 'area_4', 'area_5a', 'area_5b'],
    '3_val': ['area_1', 'area_3', 'area_6']
}

class Stanford2d3dSegmentation(SegmentationDataset):
    """Stanford2d3d Semantic Segmentation Dataset."""
    BASE_DIR = ''
    NUM_CLASS = 13
    fold = 1

    def __init__(self, root='datasets/Stanford2D3D', split='train', mode=None, transform=None, **kwargs):
        super(Stanford2d3dSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        root = os.path.join(self.root, self.BASE_DIR)
        assert os.path.exists(root), "Please put the data in {SEG_ROOT}/datasets/"
        self.images, self.masks = _get_stanford2d3d_pairs(root, self.fold, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in {}".format(os.path.join(root, split)))
        logging.info('Found {} images in the folder {}'.format(len(self.images), os.path.join(root, split)))
        with open('semantic_labels.json') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open('name2label.json') as f:
            name2id = json.load(f)
        self.colors = np.load('colors.npy')
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _val_sync_transform_resize(self, img, mask):
        short_size = self.crop_size
        img = img.resize(short_size, Image.BICUBIC)
        mask = mask.resize(short_size, Image.NEAREST)

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
        mask = _color2id(mask, img, self.id2label)
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

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ('beam', 'board', 'bookcase', 'ceiling', 'chair',
                'clutter', 'column', 'door', 'floor', 'sofa',
                'table', 'wall', 'window')

def _get_stanford2d3d_pairs(folder, fold, mode='train'):
    '''image is jpg, label is png'''
    img_paths = []
    if mode == 'train':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    elif mode == 'val':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    else:
        raise NotImplementedError
    for a in area_ids:
        img_paths += glob.glob(os.path.join(folder, '{}/data/rgb/*_rgb.png'.format(a)))
    img_paths = sorted(img_paths)
    mask_paths = [imgpath.replace('rgb', 'semantic') for imgpath in img_paths]
    return img_paths, mask_paths

def _color2id(mask, img, id2label):
    mask = np.array(mask, np.int32)
    unk = (mask[..., 0] != 0)
    mask = id2label[mask[..., 1] * 256 + mask[..., 2]]
    mask[unk] = 0
    mask[np.array(img, np.int8).sum(2) == 0] = 0
    mask -= 1  # 0->255
    return Image.fromarray(mask)




if __name__ == '__main__':
    from torchvision import transforms
    import torch.utils.data as data
     # Transforms for Normalization
    input_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225)),])
     # Create Dataset
    trainset = Stanford2d3dSegmentation(split='train', transform=input_transform)
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
