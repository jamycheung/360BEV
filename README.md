<div align="center"> 

## 360BEV: Panoramic Semantic Mapping for Indoor Bird’s-Eye View

</div>

<p align="center">
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
- [ ] 04/2023, release datasets.
- [ ] 04/2023, release code and models.

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

### 360BEV-Stanford
| Method    | Backbone | Acc    | mRecall | mPrecision | mIoU        | weights |
| :--------- | :----------: | :--------: | :---------: | :------------: | :------------: | :-------------- |
| Ours | MiT-B0       | 92.07     | 50.14      | 65.37         | 42.42          | *Coming soon...* |
| Ours | MiT-B2       | **92.80** | 53.56      | 67.72         | 45.78          | *Coming soon...* |
| Ours | MSCA-B       | 92.67     | **55.02**  | **68.02**     | **46.44**      | *Coming soon...* |

### 360BEV-Matterport
| Method    | Backbone | Acc    | mRecall | mPrecision | mIoU        | weights |
| :--------- | :----------: | :--------: | :---------: | :------------: | :------------: | :-------------- |
| Ours | MiT-B0 | 75.44 | 48.80 | 56.01 | 36.98  | *Coming soon...* |
| Ours | MiT-B2 |78.80 |59.54 |59.97  | 44.32  | *Coming soon...* |
| Ours | MSCA-B |**78.93** | **60.51** | **62.83** | **46.31** | *Coming soon...* |
## Usage

*Coming soon...*


## License

This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.