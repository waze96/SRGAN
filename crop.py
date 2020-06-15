#!/usr/bin/python
from PIL import Image
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


## ONLY WORKS WITH DIV2K DATASET BECAUSE THE NAME OF THE IMAGES!
## THIS SCRIPT SLICE AN IMAGE IN A LOT OF BOXES OF 128x128
## FROM THE DIV2K DATASET YOU OBTAIN MORE THAN 100 IMAGES FOR AN IMAGE.

## WE USE THIS SCRIPT TO SLICE IMAGES, WHEN THE IMAGES ARE CREATED IN 128x128 RESOLUTION
## WE USE THE RESIZE.PY SCRIPT TO OBTAIN THE LR IMAGES OF THIS NEW SET AND CREATE THE .MAT!

data_path = "/home/waze/Downloads/DATASETS_RESIZE/"
path = data_path + "DIV2K_train_HR/"
resultPath_y = data_path + "test/"

def crop(infile,height,width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)
            
            
        
def save_crop(img_crop, out_path, start_num,h,w):
  for k,piece in enumerate(img_crop,start_num):
      img=Image.new('RGB', (h,w), 255)
      img.paste(piece)
      path=os.path.join(out_path,"%d.png" % k)
      img.save(path)
  return k
  
  
out_num_x2=0
out_num_orig=0

dirs = os.listdir( path )
for item in dirs:
    picPath_orig = path+item
    cropLbl = crop(picPath_orig,128,128)
    out_num_orig = save_crop(cropLbl,resultPath_y,out_num_orig+1,128,128)