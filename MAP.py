#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:37:04 2020

@author: ee524
"""
import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score

def f(mean,cov,sample):
        c=1/((2*np.pi)**2)
        #print(mean)
       # print(cov)
        det=np.linalg.det(cov)
        #print(sample)
        #print(sample.transpose())
        k=np.matmul((sample.transpose()-mean.transpose()),np.linalg.inv(cov))
        k1=np.matmul(k,(sample-mean))
        #print(k1)
        ans=c*(1/m.sqrt(abs(det)))*m.exp(-0.5*k1)
       # print(ans)
       # print(c*(1/abs(det))*m.exp(-k))
        return (ans)
     
iris = datasets.load_iris()
y=iris.target
x=iris.data
#print(iris[])
#print(y)
x_train,x_test,y_train,y_test=split(x,y,test_size=0.2)
#print(len(x_train))
#prior=np.zeros(3)
#print(prior)
for i in y_train:
    if i==0:
        prior[0]+=1
    elif i==1:
        prior[1]+=1
    elif i==2:
        prior[2]+=1
prior=prior/120
#print(prior)
#p=[]

class1=np.where(y_train==0)
class2=np.where(y_train==1)
class3=np.where(y_train==2)
#print(class1,class2,class3)
mean1=x_train[class1].mean(axis=0)
mean2=x_train[class2].mean(axis=0)
mean3=x_train[class3].mean(axis=0)
#print(mean1,mean2,mean3)
cov1=np.cov(np.transpose(x_train[(class1)]))
cov2=np.cov(np.transpose(x_train[class2]))
cov3=np.cov(np.transpose(x_train[class3]))
#output=np.array(len(x_test))
output=[]
for i,j in enumerate(x_test):
    p1=(f(mean1,cov1,x_test[i]))*prior[0]
    p2=f(mean2,cov2,x_test[i])*prior[1]
    p3=f(mean3,cov3,x_test[i])*prior[2]
    p=[]
    p.append(p1)
    p.append(p2)
    p.append(p3)
   # p=np.asarray(p)
    output.append(p.index(max(p)))
    #print(out)
    #output[i]=np.where(p==max(p))
output=np.asarray(output)
print(output)
print(y_test)
#print(precision_recall_fscore_support(y_test,output))
print(confusion_matrix(y_test, output))
print(accuracy_score(y_test, output))
#print(cov1,cov2,cov3)

#print(len(p))
#print(p)
#for i,y in enumerate(y_train):
   # mean=x_train.mean(,axis=0)
    #print(mean)
