
import tensorflow as tf
#from keras.models import load_model     
#import keras
#mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt

import numpy as np


z=np.loadtxt('Genes')
#print(z)
t=np.loadtxt('Genes')

T1=np.ndarray.tolist(z)
j=np.array([z,t])
#print(j)


print(T1)


'''
a=np.column_stack((z,t))
t1=[]
t1=np.ndarray.tolist(t)
t1.append(np.ndarray.tolist(z))
t1.append(np.ndarray.tolist(z))
t1.append(np.ndarray.tolist(z))
t1.append(np.ndarray.tolist(z))
t2=np.array(t1)
print(t2)

print(np.shape(t2))
'''