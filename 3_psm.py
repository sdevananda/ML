import numpy as np
import matplotlib.pyplot as plt
import math as m
import time
def f(x,y):
    z=(-20*m.exp(-0.2*m.sqrt(0.5*((x*x)+(y*y)))))-(m.exp(0.5*(m.cos(23.14*x)+(m.cos(23.14*y)))))+m.exp(1)+20
    return z

particles=50
b_low=-10
b_up=10
w=0.05
c1=0.5
c2=1
maxitr=200
position1=np.empty(shape=(maxitr,particles))
position2=np.empty(shape=(maxitr,particles))
velocity1=np.empty(shape=(maxitr,particles))
velocity2=np.empty(shape=(maxitr,particles))
position1[0]=np.random.uniform(b_low,b_up,(particles))
position2[0]=np.random.uniform(b_low,b_up,(particles))
velocity1[0]=np.random.uniform(-20,20)
velocity2[0]=np.random.uniform(-20,20)
x=position1[0]
y=position2[0]
g_min1=0.5
g_min2=0.5
f_x=np.empty(shape=(maxitr,particles))
for i in range(0,particles):
    f_x[0][i]=(f(position1[0][i],position2[0][i]))
    if(f_x[0][i]<f(g_min1,g_min2)):
        g_min1=position1[0][i]
        g_min2=position2[0][i]

itr=0;
for i in range(1,maxitr):
    r1=np.random.uniform(0,1,particles)
    r2=np.random.uniform(0,1,particles)
    velocity1[i]=w*(velocity1[i-1])+((c1*r1*(x-position1[i-1]))+(c2*r2*(g_min1-position1[i-1])))
    #+((c1*r1*(x-position1[i-1]))+(c2*r2(g_min1-position1[i-1])))
    velocity2[i]=w*(velocity2[i-1])+((c1*r1*(y-position2[i-1]))+(c2*r2*(g_min2-position2[i-1])))
    position1[i]=position1[i-1]+velocity1[i]
    position2[i]=position2[i-1]+velocity2[i]
    for j in range(0,particles):
        f_x[i][j]=(f(position1[i][j],position2[i][j]))
        if(f_x[i][j]<f_x[i-1][j]):
            x=position1[i][j]
            y=position2[i][j]
            if(f_x[i][j]<f(g_min1,g_min2)):
                g_min1=position1[i][j]
                g_min2=position2[i][j]

X=Y=np.linspace(-10,10,100)

Z=np.empty((len(X),len(Y)))
for ele in range(0,len(X)):
    for ele2 in range(0,len(X)):
        Z[ele][ele2]=f(X[ele],Y[ele2])


for i in range(0,maxitr):
    p1=[]
    p2=[]
    #print(p1,"p2 is:",p2)
    for j in range(0,(particles)):

        p1.append(position1[i][j])
        p2.append(position2[i][j])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(X, Y, Z, levels=10)
    colour = {'black', 'red', 'green', 'blue', 'yellow'}
    plt.scatter(p1,p2)
    plt.show()
    time.sleep(0.2)


print("positon of global min",g_min1,g_min2)
print(f(g_min1,g_min2))

