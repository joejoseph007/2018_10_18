import csv, random, re, sys, os, math, numpy as np, time, subprocess, shutil
import matplotlib.pyplot as plt 
from multiprocessing import Pool
from distutils.dir_util import copy_tree
import scipy.interpolate as si


from mpl_toolkits.mplot3d import axes3d

Time=time.time()


from Constants import *
from NSGA2 import *
from Specie import *

#global Specie_List,sigma,Obj_call,Taboo_list

Specie_List=[];Taboo_list=[]
Obj_call=0


def Run_parallel(i):
	global Iter,Obj_call
	Specie_List1=Specie()#Specie_List
	Specie_List1.Read_Write(Results_Directory %(Iter,i),1)
	Specie_List1.Cost_run(Results_Directory %(Iter,i))
	Specie_List1.Read_Write(Results_Directory %(Iter,i),0)
	Obj_call+=1
	return Specie_List1.X,Specie_List1.Cost#Roundoff(Specie_List1[0].X),Roundoff(Specie_List1[0].Cost)


Current_Working_Directory=os.getcwd()

A="../Results/Generation_%i/Specie_%i"
B="../Results/Generation_%i/Population/Specie_%i"


'''
i=0
j=0
t=0
T=[]
Power=[]
thefile = open('Powers', 'a+')
while 1:
	if os.path.isdir(A%(i,j)):
		os.chdir(A%(i,j))
		X=np.loadtxt('Genes')
		Y=np.loadtxt('Power')	
		thefile.write("%.6f\n" %Y)
		os.chdir(Current_Working_Directory)
		os.chdir('../Simulations')
		np.savetxt('Genes%i'%t,X)
		np.savetxt('Power%i'%t,[Y])
		os.chdir(Current_Working_Directory)
		j+=1
		t+=1
		continue
	elif os.path.isdir(A%(i+1,0)):
		#print(i,j,t)
		T.append([i,j])
		i+=1
		j=0
		#print('here')
		continue
	else:
		break

'''

Power=np.loadtxt('Powers')
list1=[i for i in range(len(Power))]
#np.random.randint(0,len(Power))
random.shuffle(list1)


plt.plot(Power)
plt.show()
