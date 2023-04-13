<div align="center"> 

## 360BEV: Panoramic Semantic Mapping for Indoor Bird’s-Eye View

</div>

<p align="center">
<a href="https://arxiv.org/pdf/2303.11910.pdf">
    <img src="https://img.shields.io/badge/arXiv-2303.11910-red" /></a>
<a href="https://jamycheung.github.io/360BEV.html">
    <img src="https://img.shields.io/badge/Project-page-green" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" /></a>
<a href="https://github.com/jamycheung/DELIVER/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
</p>

<br />

![360BEV_paradigms](figs/360BEV_paradigms.png)


## Introduction

In this work, mapping from 360° panoramas to BEV semantics, the **360BEV** task, is established for the first time to achieve holistic representations of indoor scenes in a top-down view. Instead of relying on narrow-FoV image sequences, a panoramic image with depth information is sufficient to generate a holistic BEV semantic map. To benchmark 360BEV, we present two indoor datasets, 360BEV-Matterport and 360BEV-Stanford, both of which include egocentric panoramic images and semantic segmentation labels, as well as allocentric semantic maps.

For more details, please check our paper.

## Update

- [x] 03/2023, init repository.
- [x] 04/2023, release code and models.
- [ ] 04/2023, release datasets.

## 360BEV datasets

Prepare datasets:
- [Stanford2D3D](https://arxiv.org/abs/1702.01105)
- [Matterport3D](https://niessner.github.io/Matterport)

Extended datasets:
- 360BEV-Stanford (*Coming soon...*)
- 360BEV-Matterport (*Coming soon...*)

Data statistics:

| Dataset  | Scene  | Room | Frame | Category |
|-------------------|:----------------:|:---------------:|:----------------:|:-------------------:|
| train    |         5        |       215       |       1,040      |          13         |
| val      |         1        |        55       |        373       |          13         |
| **360BEV-Stanford**   |         6        |       270       |       1,413      |          13         |
| train    |        61        |        --       |       7,829      |          20         |
| val      |         7        |        --       |        772       |          20         |
| test     |        18        |        --       |       2,014      |          20         |
| **360BEV-Matterport** |        86        |      2,030      |      10,615      |          20         |

## 360Mapper model

![360BEV_model](figs/360BEV_model.png)

## Results and weights

### 360FV Stanford-2D3D 

| Model      | Backbone   | Input | mIoU  | weights |
| :--------- | :--------- | :---- | :---- | :------ |
| Trans4PASS | MiT-B2     | RGB   | 52.1 |         |
| CBFC       | ResNet-101 | RGB   | 52.2 |         |
| Ours       | MiT-B2     | RGB   | **54.3** | *Coming soon...* |

### 360FV-Matterport

| Model      | Backbone   | Input | mIoU  | weights |
| :--------- | :--------- | :---- | :---- | :------ |
|HoHoNet |  ResNet-101 | RGB-D | 44.85 | |
|SegFormer  |  MiT-B2 | RGB | 45.53 | |
|Ours |  MiT-B2 | RGB | **46.35** | *Coming soon...* |


### 360BEV-Stanford
| Method    | Backbone | Acc    | mRecall | mPrecision | mIoU        | weights |
| :--------- | :----------: | :--------: | :---------: | :------------: | :------------: | :-------------- |
|Trans4Map | MiT-B0 | 86.41 | 40.45 | 57.47 | 32.26 |  |
|Trans4Map | MiT-B2 | 86.53 | 45.28 | 62.61 | 36.08 |  |
| Ours | MiT-B0       | 92.07     | 50.14      | 65.37         | 42.42          | *Coming soon...* |
| Ours | MiT-B2       | **92.80** | 53.56      | 67.72         | 45.78          | *Coming soon...* |
| Ours | MSCA-B       | 92.67     | **55.02**  | **68.02**     | **46.44**      | *Coming soon...* |

### 360BEV-Matterport
| Method    | Backbone | Acc    | mRecall | mPrecision | mIoU        | weights |
| :--------- | :----------: | :--------: | :---------: | :------------: | :------------: | :-------------- |
|Trans4Map | MiT-B0 | 70.19 | 44.31 | 50.39 | 31.92  |  |
|Trans4Map | MiT-B2 | 73.28 | 51.60 | 53.02 | 36.72  |  |
| Ours | MiT-B0 | 75.44 | 48.80 | 56.01 | 36.98  | *Coming soon...* |
| Ours | MiT-B2 |78.80 |59.54 |59.97  | 44.32  | *Coming soon...* |
| Ours | MSCA-B |**78.93** | **60.51** | **62.83** | **46.31** | *Coming soon...* |
## Usage

*Coming soon...*


## License

This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.


## Citation

If you are interested in this work, please cite the following works:
```
@article{teng2023360bev,
  title={360BEV: Panoramic Semantic Mapping for Indoor Bird's-Eye View}, 
  author={Teng, Zhifeng and Zhang, Jiaming and Yang, Kailun and Peng, Kunyu and Shi, Hao and Reiß, Simon and Cao, Ke and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2303.11910},
  year={2023}
}
```
