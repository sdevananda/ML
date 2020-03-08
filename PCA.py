# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

import math as m
import glob
import scipy.linalg as la
train = [plt.imread(file) for file in glob.glob("/Users/devreddy/Desktop/IITG/2ndSEM/ML/assignment-5/Yale/train/*.pgm")]
test = [plt.imread(file) for file in glob.glob("/Users/devreddy/Desktop/IITG/2ndSEM/ML/assignment-5/Yale/test/*.pgm")]

rows,cols=train[0].shape
size=rows*cols
#print(len(train))
l=len(train)
image_matrix=np.zeros((l,size))
for i in range(len(train)):
    image_matrix[i,]=np.array(train[i].reshape(size))

mean=image_matrix.mean(axis=0)
image_mean=image_matrix-mean
cov=np.cov(image_mean)
eig=la.eig(cov)
eig_value_unsorted=eig[0]
eigen_value=np.flip(np.sort(abs(eig[0])))
#print(type(eigen_value))
cal_sum=0
#print(len(eigen_value))
for i in range(len(eigen_value)):
    cal_sum=cal_sum+eigen_value[i]
    if(cal_sum>=(0.9*sum(eigen_value))):
        k=i
        #print(k)
        eig_v1=eigen_value[i]
        break
#print(k)
#print(cal_sum)
#print(type(eigen_value))
req_eig_indices=np.argwhere(eig[0][eig[0]>=eig_v1])
#ind=
#print(req_eig)
req_eig_vector=eig[1][req_eig_indices]
req_eig_vector=req_eig_vector.reshape(k+1,150)

basis_matrix=np.dot(req_eig_vector,image_matrix)

#print(len(basis_matrix))
eigen_faces=np.zeros((k+1,rows,cols))
for i in range(len(basis_matrix)):
    eigen_faces[i]=basis_matrix[i].reshape(rows,cols)

for i in range(len(eigen_faces)):
    plt.imshow(eigen_faces[i])   
    plt.show()

test1=test[0]

test1_mat=np.array
