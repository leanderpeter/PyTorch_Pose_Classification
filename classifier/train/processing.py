import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from PIL import Image
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models



def load_data(data_folder, phase='train', train_val_split=True, train_ratio=.8):
    transform_dict = {
        'train': transforms.Compose(
            [transforms.Resize(256),
             transforms.RandomCrop(224),
             # transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'test': transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}

    data = datasets.ImageFolder(root=data_folder, transform=transform_dict[phase])
    if phase == 'train':
        if train_val_split:
            train_size = int(train_ratio * len(data))
            test_size = len(data) - train_size
            data_train, data_val = torch.utils.data.random_split(data, [train_size, test_size])

            return data_train, data_val

train, test = load_data('./dataset')

print(len(train))