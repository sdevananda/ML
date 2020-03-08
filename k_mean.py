#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:11:14 2020

@author: devreddy
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math as m
import glob
import scipy.linalg as la
train1 = cv2.imread("/Users/devreddy/Desktop/IITG/2ndSEM/ML/6th/fruits.jpg")
#plt.imshow(train)
image_array=train1.reshape((275*440,3))
image_mod=image_array
#image_array.setflags(write=1)
k=5
p=0
mean= np.random.uniform(0,255,(k,3))
#mean_new=np.zeros(())
for i in range(25):
    #p=p+1
    #print(p)
    new_mean=np.zeros((k,3))
    distance=np.zeros((121000,k))
    for i in range(len(mean)):
        distance[:,i]=np.linalg.norm(image_array-mean[i], axis=1)
    min_dist=np.zeros((121000))
    distance=distance.reshape(k,121000)
    min_dist=(distance==np.amin(distance,axis=0))
    
    for i in range(len(mean)):
        #row1=image_array[min_dist[i]]
        
        new_mean[i]=(np.mean(image_array[min_dist[i]],axis=0))
    mean_diff=np.linalg.norm(new_mean-mean)
    print(mean_diff)
#    if(mean_diff<1):
#        break
    mean=new_mean
mod_img=np.zeros((275*440,3))
for i in range(k):
    #print(min_dist[i])
   # print(np.where(min_dist[0,i]=="TRUE"))
   image_mod[np.where(min_dist[i])]=new_mean[i]
image_mod=image_mod.reshape(275,440,3)
plt.imshow(image_mod)
#print(image_mod.flags)