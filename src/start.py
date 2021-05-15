import numpy as np
import os
import cv2
from WordSegmentation import wordSegmentation, prepareImg
import json
import editdistance
from path import Path
from DataLoaderIAM import DataLoaderIAM, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import argparse
import tensorflow as tf
import subprocess as sp
import sys

for images in os.listdir('D:/SimpleHTR/data'):
    if images.endswith('.png') or images.endswith('.jpg') or images.endswith('.jpeg'):
        os.remove(os.path.join('D:/SimpleHTR/data',images)) 
        
open('D:/SimpleHTR/data/output.txt', 'w').close()
#import image
image = cv2.imread('D:/SimpleHTR/input.png')
#cv2.imshow('orig',image)
#cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)
#cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('second',thresh)
#cv2.waitKey(0)

#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
#cv2.imshow('dilated',img_dilation)
#cv2.waitKey(0)

#find contours
ctrs,hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    #cv2.imshow('segment no:'+str(i),roi)
    cv2.imwrite("D:/SimpleHTR/temp/segment_no_"+str(i)+".png",roi)
    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    #cv2.waitKey(0)

#cv2.imwrite('final_bounded_box_image.png',image)
#cv2.imshow('marked areas',image)
#cv2.waitKey(0)

os.path.join(os.path.dirname('D:/SimpleHTR/src/wordmain.py'))
tf.compat.v1.reset_default_graph()
exec(open('wordmain.py').read())
for images in os.listdir('D:/SimpleHTR/temp'):
    if images.endswith('.png') or images.endswith('.jpg') or images.endswith('.jpeg'):
        os.remove(os.path.join('D:/SimpleHTR/temp',images)) 
        
for images in os.listdir('D:/SimpleHTR/data'):
    if images.endswith('.png') or images.endswith('.jpg') or images.endswith('.jpeg'):
        os.remove(os.path.join('D:/SimpleHTR/data',images))

for images in os.listdir('D:/SimpleHTR'):
    if images.endswith('.png') or images.endswith('.jpg') or images.endswith('.jpeg'):
        os.remove(os.path.join('D:/SimpleHTR',images))




