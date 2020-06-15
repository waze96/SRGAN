#!/usr/bin/python
from PIL import Image
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime
import time

## WARNING, the MAT file is saved in Memory until it is complete. 
## You need to create small MAT file (MAX 20000 images) or your RAM will be stucked!!

##Path with the original images to obtain LR images.
data_path_X = "/home/waze/Downloads/DATASETS_RESIZE/PNG8bit/"
##Path with the original images to obtain HR images.
data_path_y = "/home/waze/Downloads/DATASETS_RESIZE/DIV2K_train_HR/"
##Path to save the images in LR.
resultPath_X = "/home/waze/Downloads/DATASETS_RESIZE/PNG8bit_32/"
##Path to save the images in HR.
resultPath_y = "/home/waze/Downloads/DATASETS_RESIZE/DIV2K_train_128/"

##Path to save the MAT file.
matfilePath = "/home/waze/Downloads/FILE_MAT/"



def resize(maxwidth, maxheight, path, resultPath):
	dirs = os.listdir( path )
	for item in dirs:
		im = Image.open(path+item)
		width, height = im.size
		i = min(maxwidth/width, maxheight/height)
		a = max(maxwidth/width, maxheight/height)
		im.thumbnail((width*a, height*a), Image.ANTIALIAS)
		width, height = im.size   # Get dimensions
		left = (width - maxwidth)/2
		top = (height - maxheight)/2
		right = (width + maxwidth)/2
		bottom = (height + maxheight)/2

		# Crop the center of the image
		im = im.crop((left, top, right, bottom))
		filename, file_extension = os.path.splitext(item)
		filename = filename + ".png"
		im.save(resultPath+filename, 'PNG')


def saveMAT(stop):
	now = datetime.datetime.now()
	start_time = time.time()
	print('Creating .mat file at {}:{}:{}'.format(now.hour, now.minute, now.second))
	dirX = os.listdir( resultPath_X )
	diry = os.listdir( resultPath_y )
	boolean = True
	mergedDir = zip(dirX, diry)
	for i,(itemX,itemy) in enumerate(mergedDir):
		immX = cv2.imread(resultPath_X + itemX)
		immY = cv2.imread(resultPath_y + itemy)

		cv2.cvtColor(immX, cv2.COLOR_BGR2RGB)
		cv2.cvtColor(immY, cv2.COLOR_BGR2RGB)

		immX = np.expand_dims(immX, axis=3)
		immY = np.expand_dims(immY, axis=3)

		if boolean:
			boolean = False
			dat = immX
			lbl = immY
		else:
			dat = np.concatenate((dat,immX),axis=3)
			lbl = np.concatenate((lbl,immY),axis=3)
		if i % (stop/10) == 0:
			print('{}% -- {} seconds'.format(i/100, (time.time() - start_time)))
		if i % stop == 0 and i != 0:
			break
	sio.savemat(matfilePath+'dataset_32to128.mat', {'X':dat, 'y':lbl})
	#sio.savemat(matfilePath+'dataset_32to128.mat', {'X':dat})
	now = datetime.datetime.now()
	print('Finished at {}:{}:{}'.format(now.hour, now.minute, now.second))



## Size for the LR images: 32x32
resize(32,32,data_path_X, resultPath_X)

## Size for the HR images: 128x128
resize(128,128,data_path_y, resultPath_y)

## Number of samples in MAT file: 10K
saveMAT(10000)
