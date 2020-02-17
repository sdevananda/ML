import numpy as np
import matplotlib.pyplot as plt
import math as m
def genRandPointsInRing(r_in,r_out,centre,n):
    r=np.random.uniform(r_in,r_out,n)
    theta=np.random.uniform(0,2*np.pi,n)
    x=centre[0]+r*np.cos(theta)
    y=centre[1]+r*np.sin(theta)
    return (x,y)

c=[0,0]
r_in=0.5
r_out=2
n=3000
points=(genRandPointsInRing(r_in,r_out,c,n))
plt.scatter(points[0],points[1])
plt.show()

n_pos=genRandPointsInRing(0,5,[0,0],250)
n_neg=genRandPointsInRing(4,8,[0,0],350)
plt.scatter(n_pos[0],n_pos[1],color='red')
plt.scatter(n_neg[0],n_neg[1],color='blue')
plt.show()

