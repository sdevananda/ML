#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:19:00 2020

@author: ee524
"""


import numpy as np
import matplotlib.pyplot as plt
import math as m
import re
import glob
import scipy.linalg as la


#import cv2
#train=plt.imread("")
train = [plt.imread(file) for file in glob.glob("./Yale/train/*.pgm")]
test = [plt.imread(file) for file in glob.glob("./Yale/test/*.pgm")]
rows,cols = train[0].shape
size=rows*cols
img_1d=[]
for i,image in enumerate(train):
    img_1d.append(train[i].reshape(size))
   # print(img_1d)
#print(type(img_1d))
x_t=(np.asmatrix(img_1d))
print(x_t.shape)
x=(x.T)
u=np.mean(x_t,axis=0)
x_k=x_t-u
#print((x_t*x_t.T).shape)
eigen=la.eig(x_t*x_t.T)
#print(eigen[0])
sorted_indices=np.argsort(eigen[0])
#print
#print(sorted_indices)
eigen_sortedvalues=[]
eigen_sortedvectors=[]
#print(sum(eigen[0]))
#print(eigen[0].shape)
for i in range(len(eigen[0])):
    #print(i)
    if(sum(np.array(eigen_sortedvalues)*np.array(eigen_sortedvalues))<(0.90*sum((eigen[0])*(eigen[0])))):
       
        eigen_sortedvalues.append(eigen[0][sorted_indices[149-i]])
        eigen_sortedvectors.append(eigen[1][sorted_indices[149-i]])
#print(eigen_sortedvalues)
#for i in range(10):
   # print(eigen_sortedvalues[i])
#print(sum(np.array(eigen_sortedvalues)*np.array(eigen_sortedvalues)))
#print(sum((eigen[0])*(eigen[0])))
print(len(eigen_sortedvalues))