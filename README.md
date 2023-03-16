## An Ensemble Learning Framework with Boundary Guided Attention for Skin Lesions Segmentation
**Paper Title**: _An Ensemble Learning Framework with Boundary Guided Attention for Skin Lesions Segmentation_   
 
by Yan Liu, Yan Yang, Xiaole Zhao and Vittor Gift Mawutor.
## Table of Contents

- [Introduction](#background)

- [Configurations](#Configurations)

- [Baseline pretrained weights](#pretrained_weights)


## Introduction
In this study, we present a novel ensemble learning framework called ES2, which sought to accurately segment skin lesions using convolution neural network (CNN), cross-scale
vision transformer and boundary knowledge. The key to ES2 is addressing skin lesion segmentation with problems such as hair cover
lesions, color variation, different sizes, and relatively low contrast resulting in ambiguous boundaries


## Configurations

Please install the following libraries, or pip install related dependencies file

1. Python 3.7.5
2. Pytorch 1.8.0 
3. torchvision 0.9.0
5. Tensorboard 2.9.1
6. opencv-python  4.6.0.66

## Baseline pretrained weights
pretrained weights are available in: https://github.com/liuyanice/BGC-Trans/tree/main/lib/backbone/pvt_v2_b2.pth

## Datasets
The pretraining dataset can be downloaded from the following URLs:

1. [isic-2016 dataset](https://pan.baidu.com/s/1p0-4w98_JIZraK3YhAxSUQ) Password is: lanq
2. [isic-2018 dataset](https://pan.baidu.com/s/18TnPCxhVKXQK1FO3NbiPEQ) Password is: lanq
3. [PH2 dataset](https://pan.baidu.com/s/1poRed9D_06pih4qRSLoEUA)  Password is: lanq  
4. [Poly-seg dataset](https://pan.baidu.com/s/1zE4MkjwKoli3Q5fNoMDVkw) Password is: lanq

## Segmentation 
In this repositories, we present BGC-Trans for segmentation task:

1. Downloading the datasets include isic-2016 dataset, isic-2017 dataset, isic-2018 dataset, PH2 dataset and Poly-seg dataset to your local path,
2. If you want to train isic-2016 dataset, please modify the path of your local isic-2016 dataset in the code of "utils/isic2016_dataset.py":
3. If you want to test PH2 dataset, please modify the path of your local PH2 dataset in the code of "utils/isic2016_dataset.py"
4. If you want to train isic-2018 dataset for 5-fold cross-validation, please modify the path of your local isic-2018 dataset in the code of "utils/isic2018_dataset.py"
5. If you want to train Poly-seg dataset, please modify the path of your local Poly-seg dataset in the code of "utils/isic2016_dataset.py"

For training, we present proposed method:
Please run the 'src/train.py' to training. (Check the super parameters as required before running in train.py)

For testing, we present proposed method:
Please run the 'src/test.py' to testing. (Check the super parameters as required before running in test.py)



## Citation
If you use the proposed framework (or any part of this code in your research) or use these processed datasets, please cite ours paper:


## Contact
If you have any query, please feel free to contact us at: 56200522@qq.com,  liuyan@my.swjtu.edu.cn
