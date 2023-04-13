"""
This module provides data loaders and transformers for popular vision datasets.
"""
# from .mscoco import COCOSegmentation
from .cityscapes import CitySegmentation
from .cityscapes13 import City13Segmentation
from .stanford2d3d import Stanford2d3dSegmentation
from .stanford2d3d_pan import Stanford2d3dPanSegmentation
from .densepass import DensePASSSegmentation
from .densepass13 import DensePASS13Segmentation
from .synpass13 import SynPASS13Segmentation
from .synpass import SynPASSSegmentation
from .stanford2d3d8 import Stanford2d3d8Segmentation
from .stanford2d3d_pan8 import Stanford2d3dPan8Segmentation
from .structured3d8 import Structured3d8Segmentation

datasets = {
    'cityscape': CitySegmentation,
    'cityscape13': City13Segmentation,
    'stanford2d3d': Stanford2d3dSegmentation,
    'stanford2d3d_pan': Stanford2d3dPanSegmentation,
    'densepass': DensePASSSegmentation,
    'densepass13': DensePASS13Segmentation,
    'synpass13': SynPASS13Segmentation,
    'synpass': SynPASSSegmentation,
    'stanford2d3d8': Stanford2d3d8Segmentation,
    'stanford2d3d_pan8': Stanford2d3dPan8Segmentation,
    'structured3d8': Structured3d8Segmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
