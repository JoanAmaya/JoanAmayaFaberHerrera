# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 19:29:50 2022

@author: soyyo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib as mpl
from tqdm import tqdm
Min, Max, N = 0.,40.,51
x = np.linspace(Min,Max,N)
y = x.copy()
h = x[1]-x[0]
def h1(y):
    return 75.
def h2(y):
    return 50.
def h3(x):
    return 100.
def h4(x):
    return 0.
def InitT():
    
    T = np.zeros((N,N))
    
    T[0,:] = h1(y)
    T[-1,:] = h2(y)
    
    T[:,0] = h3(x)
    T[:,-1] = h4(x)
    
    return T
w_optimo=2/(1+(np.pi/N))

def GetRelaxation(T, omega, Nit = int(1e5) , tolerancia = 1e-3):
    
    itmax = 0
    
    for it in tqdm(range(Nit)):
        
        dmax = 0.
        
        for i in range(1, len(x)-1):
            for j in range(1, len(y)-1):
                tmp = 0.25*(T[i+1,j]+T[i-1,j]+T[i,j+1]+T[i,j-1])
                r = omega*(tmp - T[i,j])
                
                T[i,j] += r
                
                if np.abs(r) > dmax:
                    dmax = r
        
        if np.abs(dmax) < tolerancia:
            print(it)
            itmax = it
            break
            
    return T,itmax




om=[]
vl=[]
w=1
for i in range(10):

    T = InitT()
    Tf1,it =  GetRelaxation(T,w)
    om.append(w)
    vl.append(it)
    w+=0.1

plt.plot(om,vl)
plt.title("Metodo de sobrerelajación")
plt.xlabel("omega")
plt.ylabel("iteraciones")
plt.grid()
plt.show()
plt.savefig("Punto1tareafinalJoanFaber")

