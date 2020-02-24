#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:30:59 2020

@author: ee524
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m
def genRandPointsInRing(r_in,r_out,centre,n):
    r=np.random.uniform(r_in,r_out,n)
    theta=np.random.uniform(0,2*np.pi,n)
    x=centre[0]+r*np.cos(theta)
    y=centre[1]+r*np.sin(theta)
    return (x,y)


n_pos=genRandPointsInRing(0,1,[0,0],250)
n_neg=genRandPointsInRing(7,8,[0,0],350)
plt.scatter(n_pos[0],n_pos[1],color='red')
plt.scatter(n_neg[0],n_neg[1],color='blue')
plt.show()
#print(n_pos[0])
mu_pos_x=np.average(n_pos)
mu_pos_y=np.average(n_pos[1])
sigma_pos_x=np.std(n_pos[0])
sigma_pos_y=np.std(n_pos[1])
#print(mu_pos_x,mu_pos_y)
#print(sigma_pos_x,sigma_pos_y)
k=np.linspace(0.01,10,200)
tp=np.zeros(200)
fp=np.zeros(200)
n_pos_nor_x=abs(n_pos[0]-mu_pos_x)/sigma_pos_x
n_pos_nor_y=abs(n_pos[0]-mu_pos_x)/sigma_pos_x
n_neg_nor_x=abs(n_neg[0]-mu_pos_x)/sigma_pos_x
n_neg_nor_y=abs(n_neg[0]-mu_pos_x)/sigma_pos_x
for p,i in enumerate(k):
    for j in range(len(n_pos_nor_x)):
        if(abs(n_pos_nor_x[j])<=i and abs(n_pos_nor_y[j])<=i):
            tp[p]=tp[p]+1
        if(abs(n_neg_nor_x[j])<=i and abs(n_neg_nor_y[j])<=i):
            fp[p]=fp[p]+1
#print(fp,"fn:",tp)
plt.plot((fp/350),(tp/250))
#plt.line(fp/350,tp/250)
plt.show()