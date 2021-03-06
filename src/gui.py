import tkinter as tk
from tkinter import ttk
from tkinter import * 
from tkinter import filedialog
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
import imghdr
import shutil
import matplotlib.pyplot as plt

class FilePaths:
    "filenames and paths to data"
    fnCharList = 'D:/SimpleHTR/model/charList.txt'
    fnSummary = 'D:/SimpleHTR/model/summary.json'
    fnInfer = 'D:/SimpleHTR/data/test.png'
    fnCorpus = 'D:/SimpleHTR/data/corpus.txt'

def infer(model, fnImg):
    global prob
    global now
    global xaxis
    global yaxis
    "recognize text in image provided by file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    xaxis.append(recognized[0])
    print(f'Probability: {probability[0]}')
    yaxis.append(probability[0]*100)
    prob=prob+probability[0]
    now=now+1
    apex=open("D:/SimpleHTR/data/output.txt","a")
    apex.write(recognized[0]+" ")
    apex.close()


def btnClickFunction(event=None):
    l1=Label(root,text='', bg='#F0F8FF', font=('arial', 12, 'normal'))
    l2=Label(root,text='', bg='#F0F8FF', font=('arial', 12, 'normal'))
    l1.place(x=50,y=170)
    l2.place(x=50,y=200)
    b1=Button(root, text='Recognise The Uploaded Image', bg='#F0F8FF', font=('arial', 12, 'normal'), command=recogniseFunction)
    filename = filedialog.askopenfilename()
    if(imghdr.what(filename)=='png' or imghdr.what(filename)=='jpeg' or imghdr.what(filename)=='jpg'):
        l1.config(text = 'Uploaded Image File:                                                 ',fg='green')
        l2.config(text = filename+'                                                            ')
        b1.place(x=50,y=300)
        shutil.copyfile(filename,'D:/SimpleHTR/input.png')
    else:
        l1.config(text ='This is NOT an Image File,Please Upload Again!                        ',fg='red')
        l2.config(text = filename+'                                                            ')
        b1.place(x=50,y=300)
        b1.config(state=tk.DISABLED)


    
def recogniseFunction(event=None): 
    global prob
    global now
    global xaxis
    global yaxis
    prob=0
    now=0
    xaxis.clear()
    yaxis.clear()
    exec(open('D:/SimpleHTR/src/start.py').read())
    prob=(prob/now)*100
    apex=open("D:/SimpleHTR/data/output.txt","a")
    apex.write('\r\n'+'Accuracy:'+str(prob)+' %')
    apex.close()
    outputtextFunction()
    outputgraphFunction()
    reset()
    
def outputtextFunction(event=None):
    programName = "notepad.exe"
    fileName = "D:/SimpleHTR/data/output.txt"
    sp.Popen([programName, fileName])
    
def outputgraphFunction(event=None):
    global xaxis
    global yaxis
    plt.tick_params(axis='x', labelsize=5)
    plt.bar(xaxis,yaxis, color='b')
    plt.title('Accuracy/Word Graph')
    plt.xlabel('Word Recognised')
    plt.ylabel('Accuracy(in %)')
    plt.savefig('D:/SimpleHTR/data/graph.png',dpi=1000)
    plt.show()

def reset():
    global root
    root.destroy()
    root = Tk()
    root.geometry('618x406')
    root.configure(background='#F0F8FF')
    root.title('Handwriting Recognition Tool')
    Label(root, text='Welcome to our Handwriting Recognition Tool', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=40, y=32)   
    Label(root, text='Use the Upload button below to upload an image that you want to Recognise!', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=40, y=79)
    Button(root, text='Upload an image', bg='#F0F8FF', font=('arial', 12, 'normal'), command=btnClickFunction).place(x=50, y=132)
    Button(root, text='See Previous Output Text', bg='#F0F8FF', font=('arial', 12, 'normal'), command=outputtextFunction).place(x=50,y=350)
    Button(root, text='See Previous Output Graph', bg='#F0F8FF', font=('arial', 12, 'normal'), command=outputgraphFunction).place(x=300,y=350)
    Label(root, text='Previous Accuracy:', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=300, y=250)

  
prob=0    
now=0
xaxis=list()
yaxis=list()
root = Tk()
root.geometry('618x406')
root.configure(background='#F0F8FF')
root.title('Handwriting Recognition Tool')
Label(root, text='Welcome to our Handwriting Recognition Tool', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=40, y=32)
Label(root, text='Use the Upload button below to upload an image that you want to Recognise!', bg='#F0F8FF', font=('arial', 12, 'normal')).place(x=40, y=79)
Button(root, text='Upload an image', bg='#F0F8FF', font=('arial', 12, 'normal'), command=btnClickFunction).place(x=50, y=132)
Button(root, text='See Previous Output Text', bg='#F0F8FF', font=('arial', 12, 'normal'), command=outputtextFunction).place(x=50,y=350)
Button(root, text='See Previous Output Graph', bg='#F0F8FF', font=('arial', 12, 'normal'), command=outputgraphFunction).place(x=300,y=350)

root.mainloop()


