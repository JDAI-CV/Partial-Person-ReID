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
   | Name | Academy | score | 
| - | :-: | -: | 
| Harry Potter | Gryffindor| 90 | 
| Hermione Granger | Gryffindor | 100 | 
| Draco Malfoy | Slytherin | 90 |
