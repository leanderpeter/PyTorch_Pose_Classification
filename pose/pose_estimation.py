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
from numpy import savetxt
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from torchvision import transforms
from PIL import Image

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='/home/l/pytorch_Realtime_Multi-Person_Pose_Estimation/experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)



model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

#dataset path 
path = './dataset/'

dataset = []
labels = []

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
		
		dataset = {'image': out, 'class': classes}

		'''
		cv2.imwrite('OutBlank.png',outBlank)
		cv2.imwrite('result.png',out)
		'''


print(dataset['image'], dataset['class'])
cv2.imshow(dataset['class'], dataset['image'])
cv2.waitKey(0)