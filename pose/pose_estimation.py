import os
import re
import sys
sys.path.append('../')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import model_functions
from numpy import savetxt
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from torchvision import transforms, models
from PIL import Image

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

#dataset path 
path = './dataset/'

#Dataset array
dataset = np.array([])

### Pose Estimation Model ### 
model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))

#Switch between CUDA or CPU
model = torch.nn.DataParallel(model).cuda()
#model = torch.nn.DataParallel(model)
model.float()
model.eval()




### Pose Classification Model ###
input_size = 25088
pose_model = models.vgg16(pretrained=True)
# Freeze pretrained model parameters to avoid backpropogating through them
for parameter in model.parameters():
    parameter.requires_grad = False

#Classifier settings
hidden_units = 10000

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(hidden_units, len(os.listdir(path)))),
                                        ('output', nn.LogSoftmax(dim=1))]))


print(pose_model)

for classes in os.listdir(path):
	for img in os.listdir(path + classes):

		img_path = str(path + classes + '/' + img)

		oriImg = cv2.imread(img_path) # B,G,R order

		# Processing at this point is shit cause of image distortion!
		width = 224
		height = 224
		dim = (width, height)

		# resize image
		oriImg = cv2.resize(oriImg, dim, interpolation = cv2.INTER_AREA)

		shape_dst = np.min(oriImg.shape[0:2])

		# Get results of original image

		with torch.no_grad():
		    paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
		          
		print(im_scale)

		blank = np.zeros(oriImg.shape, dtype=np.uint8)
		# blank.fill(255)

		humans = paf_to_pose_cpp(heatmap, paf, cfg)

		out = draw_humans(oriImg, humans)
		outBlank = draw_humans(blank, humans)
		
		img_set = {'image': outBlank, 'class': classes}
		#dataset.append(img_set)
		dataset = np.append(dataset, img_set)



# Training/Testing split. Biggest shit ever
np.random.shuffle(dataset)
dataset = np.array(dataset)
training, test, validation = dataset[:6], dataset[6:7], dataset[7:8]
print('Dataset lenght: ',len(dataset))
print('training lenght: ',len(training))
print('test lenght: ',len(test))
print('Validation lenght: ',len(validation))

for els in iter(training):
	# print(els)
	images, labels = els['image'], els['class']
	print(type(labels))
	print(type(images))

pose_model.classifier = classifier

# Loss function (since the output is LogSoftmax, we use NLLLoss)
criterion = nn.NLLLoss()

#Gradient Decent settings
gpu = 'cpu'
learning_rate = 0.001
epochs = 20
save_dir = 'checkpoint.pth'
arch = 'vgg'

train_loader = torch.utils.data.DataLoader(training, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validation, batch_size=32)
test_loader = torch.utils.data.DataLoader(test, batch_size=32)

for els in iter(train_loader):
	# print(els)
	images, labels = els['image'], els['class']
	print(type(labels))
	print(type(images))


'''
# Gradient descent optimizer
optimizer = optim.Adam(pose_model.classifier.parameters(), lr=learning_rate)
    
model_functions.train_classifier(pose_model, optimizer, criterion, epochs, train_loader, validate_loader, gpu)
    
model_functions.test_accuracy(pose_model, test_loader, gpu)

model_functions.save_checkpoint(pose_model, training_dataset, arch, epochs, learning_rate, hidden_units, input_size)
	
'''

#End