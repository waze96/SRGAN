#!/usr/bin/python
from PIL import Image
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

data_path = "/home/waze/Downloads/DATASETS_RESIZE/"
path = data_path + "DIV2K_train_HR/"
resultPath_X = data_path + "DIV2K_train_64/"
resultPath_y = data_path + "DIV2K_train_128/"

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

for i in range(1,801):
    picPath_orig = path+format(i,'04')+'.png'
    #cropInp = crop(picPath_orig,64,64)
    cropLbl = crop(picPath_orig,128,128)
    #out_num_x2 = save_crop(cropX2,resultPath_X,out_num_x2+1,64,64)
    out_num_orig = save_crop(cropLbl,resultPath_y,out_num_orig+1,128,128)