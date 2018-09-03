# Partial-Person-ReID
******************************************************************************************************************

**The source code**: Spatial Feature Reconstruction with Pyramid Pooling for Partial Person Re-identification 
CVPR18: [Deep Spatial Feature Reconstruction for Partial Person Re-identification: Alignment-free Approach](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_Deep_Spatial_Feature_CVPR_2018_paper.pdf), Arxiv18 

The project provides the training and testing code for partial person re-id, using [Pytorch](https://pytorch.org/)

## Instllation
*****************************************************************************************************************
It's recommended that you create and enter a python virtual environment, if versions of the packages required here conflict with yours.

Other packages are specified in `requirements.txt`

## Daset Preparation
Inspired by Houjing Huang's [person-reid-triplet-loss-baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline) project, you can follow his guidance.

## Experiment Setting:
1. Backbone: ResNet-50, `stride = 1` in the last conv block.
2. Input image size: `384 & times 192`

## Person Re-identification
### Result on Market1501
`python script/dataset/transform_cuhk03.py \
--zip_file ~/Dataset/cuhk03/cuhk03_release.zip \
--train_test_partition_file ~/Dataset/cuhk03/re_ranking_train_test_split.pkl \
--save_dir ~/Dataset/cuhk03`

   | Method | Rank-1 (Single query) | mAP | Rank-1 (Multi query)| mAP |
| - | :-: | -: |  :-: | -: | 
| Baseline | 88.18| 73.85 | 92.25 | 80.96|
| SFR | 93.04 | 81.02 | 94.84 | 85.47 |
