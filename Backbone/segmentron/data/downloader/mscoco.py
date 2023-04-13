"""Prepare MS COCO datasets"""
import os
import sys
import argparse
import zipfile

# TODO: optim code
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

from segmentron.utils import download, makedirs

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/coco')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize MS COCO dataset.',
        epilog='Example: python mscoco.py --download-dir ~/mscoco',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default=None, help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite downloaded files if set, in case they are corrupted')
    args = parser.parse_args()
    return args


def download_coco(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://images.cocodataset.org/zips/train2017.zip',
         '10ad623668ab00c62c096f0ed636d6aff41faca5'),
        ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
         '8551ee4bb5860311e79dace7e79cb91e432e78b3'),
        ('http://images.cocodataset.org/zips/val2017.zip',
         '4950dc9d00dbe1c933ee0170f5797584351d2a41'),
        # ('http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip',
        # '46cdcf715b6b4f67e980b529534e79c2edffe084'),
        # test2017.zip, for those who want to attend the competition.
        # ('http://images.cocodataset.org/zips/test2017.zip',
        #  '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'),
    ]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(path=path)


if __name__ == '__main__':
    args = parse_args()
    default_dir = os.path.join(root_path, 'datasets/coco')
    if args.download_dir is not None:
        path = args.download_dir
    else:
        path = default_dir
    if not os.path.isdir(path) or not os.path.isdir(os.path.join(path, 'train2017')) \
            or not os.path.isdir(os.path.join(path, 'val2017')) \
            or not os.path.isdir(os.path.join(path, 'annotations')):
        if args.no_download:
            raise ValueError(('{} is not a valid directory, make sure it is present.'
                              ' Or you should not disable "--no-download" to grab it'.format(path)))
        else:
            download_coco(path, overwrite=args.overwrite)

    # make symlink
    try:
        os.symlink(path, default_dir)
    except Exception as e:
        print(e)
