import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split as split
iris = datasets.load_iris()
y=iris.target
x=iris.data
#print(y)
x_train,x_test,y_train,y_test=split(x,y,test_size=0.2)
#print(len(x_train))
prior=np.zeros(3)
print(prior)
for i in y_train:
    if i==0:
        prior[0]+=1
    elif i==1:
        prior[1]+=1
    elif i==2:
        prior[2]+=1
prior=prior/120
print(prior)


