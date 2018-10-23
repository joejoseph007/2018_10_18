
import tensorflow as tf
#mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

import numpy as np
import random

#print(np.shape(y_train))
#for i in range(len(y_train)):

#	print(y_train[i])

#m = np.array(x_train)
#print(m.shape)

#print(x_train)

X1=[]
Y1=[]
i=0
while 1:
	try:
		#np.append(X,np.loadtxt('../Simulations/Genes%i'%i))
		#np.append(Y,np.loadtxt('../Simulations/Power%i'%i))
		X1.append(np.ndarray.tolist(np.loadtxt('../Simulations/Genes%i'%i)))
		Y1.append(np.ndarray.tolist(np.loadtxt('../Simulations/Power%i'%i)))
		#np.concatenate(X,)
		#np.concatenate(Y,np.loadtxt('../Simulations/Power%i'%i))
		i+=1
		#if i>2:
		#	break
		continue
	except:
		break	


#'''
#truncating excess cost values
truncate0=[]
truncate1=[]

for i in range(len(Y1)):
	if Y1[i]<-30:
		continue
	else:
		truncate0.append(X1[i])
		truncate1.append(Y1[i])

plt.plot(Y1)
plt.plot(truncate1)
plt.show()

X1=[];Y1=[]

X1=truncate0
Y1=truncate1
#'''


Numb=[i for i in range(len(Y1))]

random.shuffle(Numb)

l=int(0.9*len(Numb))

X=[]
Y=[]
for i in range(l):
	X.append(X1[i])
	Y.append(Y1[i])


X2=[]
Y2=[]
for i in range(len(Numb)-l):
	X2.append(X1[i])
	Y2.append(Y1[i])




A1=np.array(X)	
B1=np.random.rand(len(Y),1)
for i in range(len(Y)):
	B1[i]=Y[i]

Z1=A1/(A1.max()-A1.min())
Z2=B1/(B1.max()-B1.min())
Z2=Z2-Z2.min()

#'''
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  #tf.keras.layers.Dense(10, activation=tf.nn.relu),
  #tf.keras.layers.Dense(100, activation=tf.nn.relu),
  
  
  tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])



model.fit(Z1, Z2, epochs=200)




A2=np.array(X2)	
B2=np.random.rand(len(Y2),1)
for i in range(len(Y2)):
	B2[i]=Y2[i]
Z1=A2/(A2.max()-A2.min())
Z2=B2/(B2.max()-B2.min())
Z2=Z2-Z2.min()

#Z2=B2/B2.max()


#print(model.evaluate(Z1, Z2))

Answer=model.predict(Z1)
print(Z2)
print(Answer)
print(len(Z2),len(Answer))


percentage=(Z2-Answer)/Z2

plt.plot(Z2)
plt.plot(Answer)
plt.show()

#print(Z1)

#'''




'''
import numpy as np

import tensorflow as tf
mnist = tf.keras.datasets.mnist

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))


'''