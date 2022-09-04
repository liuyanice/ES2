import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import cv2
import sys
import argparse
import time
import matplotlib.pyplot as plt;
#import dataloader
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
from time import *


file_path = r"root\BGC-Trans\dataset\ISIC-2018_Training_Part2_GroundTruth\ISIC-2018_Training_Part2_GroundTruth"
path_list = os.listdir(file_path)  
print(path_list)
path_name = []  


def saveList(pathName):
    for file_name in pathName:
        with open("Image.txt", "a") as f:
            f.write("\""+file_name.split("_")[1] + "\""+ ","+" ")


def dirList(path_list):
    for i in range(0, len(path_list)):
        path = os.path.join(file_path, path_list[i])
    if os.path.isdir(path):
        saveList(os.listdir(path))


dirList(path_list)
saveList(path_list)

if __name__ == '__main__':
   
    test_list = glob.glob(r"root\BGC-Trans\test\22\*")
    for path in test_list:
        im = Image.open(path)
        transF = os.path.splitext(path) 
        print(path.split(".")[0])