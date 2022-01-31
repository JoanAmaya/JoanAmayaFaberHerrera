# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:28:14 2022

@author: soyyo
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm 
import matplotlib.animation as anim
class Particle3d():
    
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
        self.radius = radius
        self.Id = Id
        
    # Method
    def Evolution3d(self,i):
        
        self.SetPosition3d(i,self.r)
        self.SetVelocity3d(i,self.v)
        

        self.r += self.dt * self.v
        self.v += self.dt * self.a
    
    def CheckWallLimits3d(self,limits,dim=3):
        
        for i in range(dim):
            
            if self.r[i] + self.radius > limits[i]:
                self.v[i] = - self.v[i]
            if self.r[i] - self.radius < - limits[i]:
                self.v[i] = - self.v[i]
    
    # Setters
    
    def SetPosition3d(self,i,r):
        self.rVector[i] = r
        
    def SetVelocity3d(self,i,v):
        self.vVector[i] = v
        
    # Getters  
    def GetPositionVector3d(self):
        return self.rVector
    
    def GetRPositionVector3d(self):
        return self.RrVector 
    

    def GetVelocityVector3d(self):
        return self.vVector
    
    def GetR(self):
        return self.radius
    
    def ReduceSize(self,factor):
        
        self.RrVector = np.array([self.rVector[0]]) # initial condition
        
        
        for i in range(1,len(self.rVector)):
            if i%factor == 0:
                self.RrVector = np.vstack([self.RrVector,self.rVector[i]])
dt = 0.01
tmax = 30
t = np.arange(0,tmax+dt,dt)

def GetParticles3d(NParticles,Limit,Velo,Dim=3,dt=0.1):
    
    Particles_ = []
    
    for i in range(NParticles):
        
        x0 = np.random.uniform( -Limit+1.0, Limit-1.0, size=Dim )
        v0 = np.random.uniform( -Velo, Velo, size=Dim)
        a0 = np.zeros(Dim)
        
        p = Particle3d(x0,v0,a0,t,1.,1.0,i)
        
        Particles_.append(p)
        
    return Particles_
Limits = np.array([10.,10.,10.])
def RunSimulation3d(t,NParticles = 100, Velo = 6):
    
    Particles = GetParticles3d(NParticles,Limits[0],Velo = Velo,dt=dt)
    
    for it in tqdm(range(len(t))): # Evolucion temporal
        for i in range(len(Particles)):
            
            Particles[i].CheckWallLimits3d(Limits)
            Particles[i].Evolution3d(it)
        
        
    return Particles
Particles = RunSimulation3d(t,100,Velo=6)

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
ax = fig.add_subplot(111, projection='3d')


def init3d():
    ax.set_xlim(-Limits[0],Limits[0])
    ax.set_ylim(-Limits[1],Limits[1])
    ax.set_zlim(-Limits[2],Limits[2])

def Update(i):
    
    plot = ax.clear()
    init3d()
    plot = ax.set_title(r'$t=%.2f \ seconds$' %(redt[i]), fontsize=15)
    ax.set_xlabel('X[m]')
    ax.set_ylabel('Y[m]')
    ax.set_zlabel('Z[m]')
    
    for p in Particles:
        x = p.GetRPositionVector3d()[i,0]
        y = p.GetRPositionVector3d()[i,1]
        z = p.GetRPositionVector3d()[i,2]
        
        
        
        ax.scatter(x, y, z, c="black", s=5)

        
    return plot

Animation = anim.FuncAnimation(fig,Update,frames=len(redt),init_func=init3d)

Writer = anim.writers['ffmpeg']
writer_ = Writer(fps=60, metadata=dict(artist='JoanFaber'))
Animation.save('Gas3d.mp4', writer=writer_)
 
