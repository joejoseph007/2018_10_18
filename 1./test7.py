import csv, random, re, sys, os, math, numpy as np, time, subprocess, shutil
import matplotlib.pyplot as plt 
from multiprocessing import Pool
from distutils.dir_util import copy_tree
#import scipy.interpolate as si
from scipy import interpolate

from mpl_toolkits.mplot3d import axes3d

Time=time.time()

def bspline(i):
    # Split into new function
    os.chdir("../Simulations")
    Genes=np.loadtxt('Genes%i'%i) 
    os.chdir("../1.")
    ctr = Genes
    #ltr = np.array(self.uPoint)

    x = ctr[:,0]
    y = ctr[:,1]

    #x1 = ltr[:,0]
    #y1 = ltr[:,1]


    l=len(x)
    t=np.linspace(0,1,l-2,endpoint=True)
    t=np.append([0,0,0],t)
    t=np.append(t,[1,1,1])

    tck=[t,[x,y],3]
    #lck=[t,[x1,y1],3]
    u3=np.linspace(0,1,(max(l*2,70)),endpoint=True)
    out = interpolate.splev(u3,tck) 
    #out1 = interpolate.splev(u3,lck) 
    
    X1=np.array(out[0])
    #X2=np.array(out1[0])
    #X3=X1[: : -1]

    #X=np.concatenate((X3,X2), 0)
    plotX = X1

    Y1=np.array(out[1])
    #Y2=np.array(out1[1])
    #Y3=Y1[: : -1]

    #Y=np.concatenate((Y3,Y2), 0)
    plotY = Y1

    #STL_Gen(X,Y,g,s)
    print("Airfoil created")

    os.chdir("../Images")
    savefig([plotX,plotY],Genes,i)
    os.chdir("../1.")
    #return plotX,plotY


def savefig(Points,Cpoints,i):   #display using matplotlib

    plt.plot(Cpoints[:,0],Cpoints[:,1],'k--',label='Control polygon',marker='o',markerfacecolor='red')
    
    #plt.plot(self.lPoint[:,0],self.lPoint[:,1],'k--',label='Control polygon',marker='o',markerfacecolor='green')
    
    plt.plot(Points[0],Points[1],'b',linewidth=2.0,label='B-spline curve')
    
    plt.legend(loc='best')
    plt.axis('equal')
    plt.axis([150, 350, -170, 170])
    plt.title('Cubic B-spline curve evaluation')
    plt.savefig('Blade%i.svg'%i, bbox_inches = "tight")
    #copyfile('airfoil_%i-%i.png', 'Results_XFoil/Generation_%i/Specie_%i/airfoil_%i-%i.png'%(self.generation,self.specie,self.generation,self.specie))
    #plt.savefig('Results_XFoil/Generation_%i/Specie_%i/airfoil_%i-%i.png'%(self.generation,self.specie,self.generation,self.specie), bbox_inches = 'tight')
    plt.close()



y = Pool(25)
result = y.map(bspline,range(612))
y.close()
y.join()    
