# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 21:09:12 2022

@author: soyyo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm 
import matplotlib.animation as anim

class Particle():
    
    # init
    def __init__(self, r0,v0,a0,t,m,radius,Id):
        
        self.dt  = t[1] - t[0]
        
        self.r = r0
        self.v = v0
        self.a = a0
        
        
        self.rVector = np.zeros( (len(t),len(r0)) )
        
        self.vVector = np.zeros( (len(t),len(v0)) )
        self.aVector = np.zeros( (len(t),len(a0)) )
        
        self.m = m
        self.Kineticenergy= (1/2)*m*(v0**2)
        self.Potencialenergy= m*(9.8)*((r0-radius)+20)

        self.radius = radius
        self.Id = Id
        self.KVector = np.zeros( (len(t),len(r0)) )
        self.PVector = np.zeros( (len(t),len(r0)) )
    # Method
    def Evolution(self,i):
        
        self.SetPosition(i,self.r)
        self.SetVelocity(i,self.v)
        self.SetK(i,self.Kineticenergy)
        self.SetP(i,self.Potencialenergy)
        
       # print(self.r)
        
        # Euler method
        self.r += self.dt * self.v
        self.v += self.dt * self.a
        self.Kineticenergy =(1/2)*self.m*(self.v**2)
        self.Potencialenergy = self.m*9.8*((self.r-self.radius)+20)
    
    def CheckWallLimits(self,i,limits,dim=2):
        
        if self.v[1]<0:
            if self.r[1] + self.radius > limits[1]:
                 self.v[1] = - self.v[1]*0.9
                 self.vVector[i] = self.v
            if self.r[1] - self.radius < - limits[1]:
                self.v[1] = - (self.v[1]*0.9)
                self.vVector[i] = self.v
        
        
        if self.r[0] + self.radius > limits[0]:
             self.v[0] = - self.v[0]*0.9
             self.vVector[i] = self.v
        if self.r[0] - self.radius < - limits[0]:
            self.v[0] = - (self.v[0]*0.9)
            self.vVector[i] = self.v        
    # Setters
    
    def SetPosition(self,i,r):
        self.rVector[i] = r
        
    def SetVelocity(self,i,v):
        self.vVector[i] = v
        
    def SetK(self,i,k):
        self.KVector[i] = k
    
    def SetP(self,i,p):
        self.PVector[i] = p     
        
    # Getters  
    def GetPositionVector(self):
        return self.rVector
    
    def GetKVector(self):
        return self.KVector
    
    def GetPVector(self):
        return self.PVector
    
    def GetRPositionVector(self):
        return self.RrVector 

    def GetRVelocityVector(self):
        return self.RvVector         

    def GetVelocityVector(self):
        return self.vVector
    
    def GetR(self):
        return self.radius
    

    
    def ReduceSize(self,factor):
    
            self.RrVector = np.array([self.rVector[0]])
            self.RvVector = np.array([self.vVector[0]])# initial condition
    
    
            for i in range(1,len(self.rVector)):
                if i%factor == 0:
                    self.RrVector = np.vstack([self.RrVector,self.rVector[i]])
                    self.RvVector = np.vstack([self.RvVector,self.vVector[i]])     
                
                
dt = 0.01
tmax = 30
t = np.arange(0,tmax+dt,dt)

x_0=np.array([-15.0,5.0])
v_0=np.array([1.0,0.0])
a_0=np.array([0.0,-9.8])

Particles_ = []
p = Particle(x_0,v_0,a_0,t,1.,1.0,0)
Particles_.append(p)

Limits= np.array([20.,20.])

def RunSimulation(t):
    
    for it in tqdm(range(len(t))):
        Particles_[0].CheckWallLimits(it,Limits)
        Particles_[0].Evolution(it)
        
    return Particles_     
                
Particles = RunSimulation(t)

def ReduceTime(t,factor):
    
    for p in Particles:
        p.ReduceSize(factor)
        
    Newt = []
    
    for i in range(len(t)):
        if i%factor == 0:
            Newt.append(t[i])
            
    return np.array(Newt)

redt = ReduceTime(t,10)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)


def init():
    ax.set_xlim(-Limits[0],Limits[0])
    ax.set_ylim(-Limits[1],Limits[1])

def Update(i):
    
    plot = ax.clear()
    init()
    plot = ax.set_title(r'$t=%.2f \ seconds$' %(redt[i]), fontsize=15)
    
    for p in Particles:
        x = p.GetRPositionVector()[i,0]
        y = p.GetRPositionVector()[i,1]
        
        vx = p.GetRVelocityVector()[i,0]
        vy = p.GetRVelocityVector()[i,1]

        
        
        circle = plt.Circle( (x,y), p.GetR(), color='k', fill=False)
        plot = ax.add_patch(circle)
        plot = ax.arrow(x,y,vx,vy,color='r',head_width=0.5)
        
    return plot



Kinetic= Particles[0].GetKVector()
Potencial= Particles[0].GetPVector()
Mechanical0= np.sqrt(Kinetic[:,0]**2+Kinetic[:,1]**2)
mec=Potencial[:,1]+Mechanical0
print(Particles[0].GetVelocityVector())



Animation = anim.FuncAnimation(fig,Update,frames=len(redt),init_func=init)

Writer = anim.writers['ffmpeg']
writer_ = Writer(fps=60, metadata=dict(artist='JoanFaber'))
Animation.save('FallingBall.mp4', writer=writer_)



dt2 = 0.01
tmax2 = 100
t2 = np.arange(0,tmax2+dt2,dt2)

x_02=np.array([-15.0,5.0])
v_02=np.array([1.0,0.0])
a_02=np.array([0.0,-9.8])

Particles2_ = []
p2 = Particle(x_02,v_02,a_02,t2,1.,1.0,0)
Particles2_.append(p2)

Limits2= np.array([20.,20.])

def RunSimulation2(t):
    
    for it in tqdm(range(len(t))):
        Particles2_[0].CheckWallLimits(it,Limits2)
        Particles2_[0].Evolution(it)
        
    return Particles2_     
                
Particles2 = RunSimulation2(t2)




Kinetic= Particles2[0].GetKVector()
Potencial= Particles2[0].GetPVector()
Mechanical0= np.sqrt(Kinetic[:,0]**2+Kinetic[:,1]**2)
mec1=Potencial[:,1]+Mechanical0

Tiempo_total= len(mec1[mec1>0.7])*0.01



print(Tiempo_total)
plt.figure(figsize=(7,7))
plt.plot(t,mec)
plt.xlabel("Tiempo")
plt.ylabel("Energía mecánica")
plt.title("el tiempo que tarda en dejar de rebotar es "+str(Tiempo_total))
plt.savefig("JoanFaber Ej1 energía mecánica.png")


