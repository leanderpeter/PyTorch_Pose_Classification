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

from lib.network.rtpose_vgg import get_model

image_dir = './test_3SZ.jpg'
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


''' Load estimation model
'''

est_model = get_model('vgg19')     
est_model.load_state_dict(torch.load('pose_model.pth'))
est_model = torch.nn.DataParallel(est_model).cuda()
est_model.float()
est_model.eval()


probabilities, classes, indices, out, outFS = model_functions.predict(image_dir, model2, est_model, topk, gpu)

print('Pose is:', classes[0], 'with a probabilitie of: ', round((probabilities[0]*100), 2),'%')

img = outFS

# write result on image

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,50)
fontScale              = 1
fontColor              = (0,0,255) #Blue Green Red
lineType               = 2

result = 'Direction: ' + classes[0]

cv2.putText(img, str(result), 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

#Display the image
cv2.imshow("img",img)

#Save image
cv2.imwrite("out.jpg", img)

cv2.waitKey(0)