import os
import cv2

class preprocessing:

	def __init__(self):
		super().__init__()
		self._dataset_path = ''
		self._model = None
		self._processed_dataset = []

	def process_dataset(self, _dataset_path):

		# Get all Labels and transform into numerical values (needed for model)
		tmp = []
		labels = []
		# get labels
		for lbl in os.listdir(_dataset_path):
			tmp.append(lbl)
		# put labels in datastructure
		for i in range(len(tmp)):
			labels.append({tmp[i]: i})
		print(labels)


	def pose_estimation():
		pass

	def img_read(self, img_path, size=224):
		# read image
		oriimage = cv2.imread(img_path)

		# rescale image to the imput size factor (standard = 224)
		scale_factor = size/oriimage.shape[0]
		newx,newy = oriimage.shape[1]*scale_factor,oriimage.shape[0]*scale_factor #new size (w,h)
		newimage = cv2.resize(oriimage,(int(newx),int(newy)))

		return newimage

