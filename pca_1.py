import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd
import scipy.linalg as la

c=(100,100)
#print((c[0]))
b=20
a=2*b
n=1000
theta=np.random.uniform(0,2*np.pi,n)
#print(type(theta))
r_x=np.random.uniform(0,a,n)
r_y=np.random.uniform(0,b,n)
x=r_x*np.cos(theta)
y=r_y*np.sin(theta)
x_new=c[0]+(x*np.cos((np.pi/4))+y*np.sin(np.pi/4))
y_new=c[1]+((x*np.sin(np.pi/4))+-y*np.cos(np.pi/4))
major_x=np.linspace(-a,a,1000)
major_y=0
minor_x=0
minor_y=np.linspace(-b,b,100)
x_major=c[0]+major_x*np.cos((np.pi/4))+major_y*np.sin(np.pi/4)
y_major=c[1]+(major_x*np.sin(np.pi/4))+-major_y*np.cos(np.pi/4)
x_minor=c[0]+minor_x*np.cos((np.pi/4))+minor_y*np.sin(np.pi/4)
y_minor=c[1]+(minor_x*np.sin(np.pi/4))+-minor_y*np.cos(np.pi/4)
plt.scatter(x_new,y_new)
plt.plot(x_major,y_major,'g')
plt.plot(x_minor,y_minor,'g')
#plt.scatter(x,y)

#print(type(x_new))
data=np.vstack((x_new,y_new))
cov=np.cov(data)
#print(cov)
eig_val=(la.eig(cov)[0])
#print(eig_val)
eig_vect=(la.eig(cov)[1])
#print(eig_vect)
e_max=np.amax(eig_val)
e_min=np.amin(eig_val)
e_max_index=np.where(eig_val==np.amax(eig_val))
e_min_index=np.where(eig_val==np.amin(eig_val))
max_vect=eig_vect[e_max_index]*50
min_vect=eig_vect[e_min_index]*10
print(max_vect)
print(10*max_vect[:,1])
plt.quiver(*c,min_vect[:,1],min_vect[:,0],color="r")
plt.quiver(*c,max_vect[:,1],max_vect[:,0],color="r")
plt.show()


#print(np.var(x_new))

