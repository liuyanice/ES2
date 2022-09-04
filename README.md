#BGC-Trans: Cross-scale Transformers with Boundary Guided Attention for Medical Image Segmentation
**Paper Title**: _BGC-Trans: Cross-scale Transformers with Boundary Guided Attention for Medical Image Segmentation_

## Table of Contents

- [Background and Abstract](#background)

- [Configurations](#Configurations)

- [Pretrained weights](#pretrained_weights)


## Background and Abstract
Although transformers have been widely used for medical image segmentation due to their superiority in global feature learning, these transformer-based methods suffer from two key issues. One is the inaccurate boundary prediction of lesion segmentation. The other is that these transformers lack an effective fusion mechanism for different-level features. Different form  existing transformer-based methods, we propose a novel transformer model called BGC-Trans, which based on boundary guided attention and cross-scale combination. BGC-Trans has three advantages. The first advantage is that we design a cross-scale transformers branch (CSTB) to extract the semantic information of high-level features, so as to better learn the context information for lesion image. The second advantage is we introduce a pixel level encoder branch (PLEB), which uses the boundary guided attention module to fuse the global and pix-level features to obtain a more accurate segmentation image. The third advantage isthat considering the problems of under-sampling and over-sampling in most lesion segmentation models, which lead to the loss of detail and edge information, a boundary compensation module (BCM) is specially designed. We evaluated BGC-Trans on three skin lesion datasets, ISIC-2016, ISIC-2018 and PH$^{2}$. Moreover, in order to confirm the generalization of our model, we also carried out extensive experiments on polyp datasets. Experiments demonstrate BGC-Trans is more robust to various challenging situations (\textsl{e}.\textsl{g}. hair coverage, varying shapes, and small objects) than state-of-the-art models.

<div align=center>
<img src="https://github.com/liuyanice/BGC-Trans/blob/main/BGC.svg" width="500px">
</div>

## Configurations

Please install the following libraries, or pip install related dependencies file

1. python 3.7.5
2. torch 1.8.0 + cu111
3. albumentations 1.2.0
4. tqdm 4.19.9
5. tensorboard 2.9.1
6. opencv-python  4.6.0.66

## baseline pretrained weights
pretrained weights are available in: https://github.com/liuyanice/BGC-Trans/tree/main/lib/backbone/pvt_v2_b2.pth

## Datasets
The pretraining dataset can be downloaded from the following URLs:

1. [isic-2016 dataset](https://pan.baidu.com/s/1p0-4w98_JIZraK3YhAxSUQ) Password is: lanq
2. [isic-2018 dataset](https://pan.baidu.com/s/18TnPCxhVKXQK1FO3NbiPEQ) Password is: lanq
3. [PH2 dataset](https://pan.baidu.com/s/1poRed9D_06pih4qRSLoEUA)  Password is: lanq  
4. [Poly-seg dataset](https://pan.baidu.com/s/1zE4MkjwKoli3Q5fNoMDVkw) Password is: lanq

## Segmentation 
In this repositories, we present BGC-Trans for segmentation task:

1. Download the datasets include isic-2016 dataset, isic-2017 dataset, isic-2018 dataset, PH2 dataset and Poly-seg dataset to your local path,
2. If you want to train isic-2016 dataset, please modify the path of your local isic-2016 dataset in the code of "utils/isic2016_dataset.py":
3. If you want to test PH2 dataset, please modify the path of your local PH2 dataset in the code of "utils/isic2016_dataset.py"
4. If you want to train isic-2018 dataset for 5-fold cross-validation, please modify the path of your local isic-2018 dataset in the code of "utils/isic2018_dataset.py"
5. If you want to train Poly-seg dataset, please modify the path of your local Poly-seg dataset in the code of "utils/isic2016_dataset.py"

For training, we present proposed method:
Please run the 'src/train.py' to training. (Set the super parameters as required before running in train.py)

For testing, we present proposed method:
Please run the 'src/test.py' to testing. (Set the super parameters as required before running in test.py)

### Results
The results of the proposed framework are presented in the terminal.

Visualization:

<div align=center>
<img src="https://github.com/liuyanice/BGC-Trans/blob/main/compare.svg" width="750px">
</div>
<br/>

<div align=center>
<img src="https://github.com/liuyanice/BGC-Trans/blob/main/polyp1.svg" width="750px">
</div>
<br/>

<div align=center>
<img src="https://github.com/liuyanice/BGC-Trans/blob/main/polyp2.svg" width="750px">
</div>
<br/>

Semgentation Results:

<div align=center>
<img src="https://github.com/liuyanice/BGC-Trans/blob/main/PolypDice.svg" width="750px">
</div>


## Citation
If you use the proposed framework (or any part of this code in your research) or use these datasets and utils tool, please cite ours paper:


## Contact
If you have any query, please feel free to contact us at: 56200522@qq.com,  liuyan@my.swjtu.edu.cn and yyang@swjtu.edu.cn
