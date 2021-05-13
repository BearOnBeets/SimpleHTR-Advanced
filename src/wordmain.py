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
import sys


def main():
  
    # read input images from 'in' directory
    imgFiles = os.listdir('D:/SimpleHTR/temp/')
    for (i,f) in enumerate(imgFiles):
        print('Segmenting words of sample %s'%f)
        
        # read image, prepare it by resizing it to fixed height and converting it to grayscale
        img = prepareImg(cv2.imread('D:/SimpleHTR/temp/%s'%f), 50)
        
        # execute segmentation with given parameters
        # -kernelSize: size of filter kernel (odd integer)
        # -sigma: standard deviation of Gaussian function used for filter kernel
        # -theta: approximated width/height ratio of words, filter function is distorted by this factor
        # - minArea: ignore word candidates smaller than specified area
        res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
        
        # write output to 'out/inputFileName' directory
        '''if not os.path.exists('D:/SimpleHTR/out/%s'%f):
            os.mkdir('D:/SimpleHTR/out/%s'%f)'''
        
        # iterate over all segmented words
        print('Segmented into %d words'%len(res))
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('D:/SimpleHTR/data/test.png', wordImg) # save word
            cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
            os.path.join(os.path.dirname('D:/SimpleHTR/src/main.py'))
            tf.compat.v1.reset_default_graph()
            exec(open('main.py').read())
        
        # output summary image with bounding boxes around words
        #cv2.imwrite('D:/SimpleHTR/data/summary.png', img)

        apex = open("D:/SimpleHTR/data/output.txt","a")
        apex.write("\n")
        apex.close()
        

if __name__ == '__main__':
    main()