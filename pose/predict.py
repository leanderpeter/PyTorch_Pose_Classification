import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import model_functions
import processing

import json
import argparse
import cv2

image_dir = './test_stop.png'
checkpoint = 'checkpoint.pth'
topk = 2
# class_to_name_dict = {0:'Stop',1:'Left',2:'Right'}
class_to_name_dict = {'Stop':0,'Left':1,'Right':2}
# class_to_name_dict =[0,1,2]
gpu = 'cuda'

model2 = model_functions.load_checkpoint(checkpoint)
print(model2)

checkpoint = torch.load(checkpoint)

image = cv2.imread(image_dir)
image = image.transpose((2, 0, 1))

probabilities, classes, indices = model_functions.predict(image_dir, model2, topk, gpu)

print('Pose is:', classes[0], 'with a probabilitie of: ', round((probabilities[0]*100), 2),'%')