import os
import glob
import json
import torch
import random
import torch.nn as nn
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
import cv2

import albumentations as A
from sklearn.model_selection import KFold


def norm01(x):
    return np.clip(x, 0, 255) / 255

# 5 fold cross-validation
seperable_indexes = json.load(open(r'root/BGC-Trans/utils/new_json/data_split.json', 'r'))



class myDataset(data.Dataset):
    def __init__(self, fold, split, size=352, aug=False):
        super(myDataset, self).__init__()
        self.split = split
        # load images, label
        self.image_paths = []
        self.label_paths = []
        self.dist_paths = []

        indexes = os.listdir(
            r'root/BGC-Trans/isic2018/Image')
        valid_indexes = [
            'ISIC_' + i + '.jpg' for i in seperable_indexes[str(fold)]
        ]
        train_indexes = list(filter(lambda x: x not in valid_indexes, indexes))
        print(len(indexes), len(train_indexes), len(valid_indexes))
        print('Fold {}: train: {} valid: {}'.format(fold, len(train_indexes),
                                                    len(valid_indexes)))

        root_dir = r'root/BGC-Trans/isic2018/'

        if split == 'train':
            self.image_paths = [
                f'{root_dir}/Image/{_id}' for _id in train_indexes
            ]
            self.label_paths = [
                f'{root_dir}/Label/{_id}' for _id in train_indexes
            ]
        elif split == 'valid':
            self.image_paths = [
                f'{root_dir}/Image/{_id}' for _id in valid_indexes
            ]
            self.label_paths = [
                f'{root_dir}/Label/{_id}' for _id in valid_indexes
            ]

        print('Loaded {} frames'.format(len(self.image_paths)))
        self.num_samples = len(self.image_paths)
        self.aug = aug
        self.size = size

        p = 0.5
        self.transf = A.Compose([
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            #             A.RandomBrightnessContrast(p=p),
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_data = cv2.imread(self.label_paths[index], cv2.IMREAD_GRAYSCALE)
        label_data = np.array(cv2.resize(label_data, (self.size, self.size), cv2.INTER_NEAREST))
        label_data = label_data / 255. > 0.5
        image_data = np.array(cv2.resize(image_data, (self.size, self.size), cv2.INTER_LINEAR))
        if self.aug and self.split == 'train':
            mask = np.concatenate([label_data[..., np.newaxis].astype('uint8')], axis=-1)
            tsf = self.transf(image=image_data.astype('uint8'), mask=mask)
            image_data, mask_aug = tsf['image'], tsf['mask']
            label_data = mask_aug[:, :, 0]
        image_data = norm01(image_data)
        label_data = np.expand_dims(label_data, 0)

        image_data = torch.from_numpy(image_data).float()
        label_data = torch.from_numpy(label_data).float()
        image_data = image_data.permute(2, 0, 1)
        return {
            'image_path': self.image_paths[index],
            'label_path': self.label_paths[index],
            'image': image_data,
            'label': label_data
        }

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    from tqdm import tqdm
    dataset = myDataset(fold='0', split='train', aug=True)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=8,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)
    import matplotlib.pyplot as plt
    for d in dataset:
        print(d['image'].shape, d['image'].max())
        image = d['image'].permute(1, 2, 0).cpu()
        plt.imshow(image)
        plt.show()
        break
