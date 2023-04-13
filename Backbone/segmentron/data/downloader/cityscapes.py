"""Prepare Cityscapes dataset"""
import os
import sys
import argparse
import zipfile

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

from segmentron.utils import makedirs, check_sha1

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/citys')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize ADE20K dataset.',
        epilog='Example: python prepare_cityscapes.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args


def download_city(path, overwrite=False):
    _CITY_DOWNLOAD_URLS = [
        ('gtFine_trainvaltest.zip', '99f532cb1af174f5fcc4c5bc8feea8c66246ddbc'),
        ('leftImg8bit_trainvaltest.zip', '2c0b77ce9933cc635adda307fbba5566f5d9d404')]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for filename, checksum in _CITY_DOWNLOAD_URLS:
        if not check_sha1(filename, checksum):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(filename))
        # extract
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path=path)
        print("Extracted", filename)


if __name__ == '__main__':
    args = parse_args()
    makedirs(os.path.expanduser('~/.torch/datasets'))
    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)
    else:
        download_city(_TARGET_DIR, overwrite=False)
