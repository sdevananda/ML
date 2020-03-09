#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:56:18 2020

@author: ee524
"""

import numpy as np
import matplotlib.pyplot as plt

import math as m
import glob
import scipy.linalg as la
train = [plt.imread(file) for file in glob.glob("./Yale/train/*.pgm")]
test = [plt.imread(file) for file in glob.glob("./Yale/test/*.pgm")]

rows,cols=train[0].shape
size=rows*cols
#print(len(train))
l=len(train)
image_matrix=np.zeros((l,size))
for i in range(len(train)):
    image_matrix[i,]=np.array(train[i].reshape(size))
norm=np.linalg.norm(image_matrix,axis=1)
image_nor=np.zeros((image_matrix.shape))
for i in range(len(train)):
    image_nor[i]=image_matrix[i]/norm[i]
k=10
init_lab=np.random.randint(0,k,size=l)
means=np.zeros((k,size))
for i in range(k):
    #print(np.mean(image_nor[init_lab==i],axis=0))
   means[i]=np.mean(image_nor[init_lab==i],axis=0)
dot_mean=np.zeros((k,len(train)))
for i in range(len(means)): 
    dot_mean[i]=(np.linalg.norm(image_nor-means[i],axis=1) )   
min_dist=np.zeros((121000,5))
min_dist=(dot_mean==np.amin(dot_mean,axis=0))
new_lab=np.empty((len(train)))
for i in range(k):
    new_lab[min_dist[i]]=i
