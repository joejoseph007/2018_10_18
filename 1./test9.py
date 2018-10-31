
import tensorflow as tf
#from keras.models import load_model     
#import keras
#mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt

import numpy as np



def Deep_neural_net(Z1,Z2):
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten()])
    #model =tf.keras.layers.Flatten()
    model.add(tf.keras.layers.Dense(1000,kernel_initializer='uniform',activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1000,kernel_initializer='uniform',activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1000,kernel_initializer='uniform',activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1000,kernel_initializer='uniform',activation=tf.nn.relu))
    
    model.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))
    
    '''
    model.add(tf.keras.layers.Embedding(10,1))
    model.add(tf.keras.layers.LSTM(128,dropout=0.2,recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    '''
    
    
    
    '''
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation=tf.nn.relu),
    tf.keras.layers.Dense(1000, activation=tf.nn.relu),
    tf.keras.layers.Dense(1000, activation=tf.nn.relu),
    tf.keras.layers.Dense(1000, activation=tf.nn.relu),
    
    #tf.keras.layers.Dense(10, activation=tf.nn.relu),
    #tf.keras.layers.Dense(100, activation=tf.nn.relu),
    
    
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    '''
    model.compile(optimizer='adamax',
                loss='binary_crossentropy',
                metrics=['mae'])

    '''
    model.compile(optimizer='adamax',
                loss='mse',
                metrics=['mae'])
    '''


    model.fit(Z1, Z2, epochs=200)
    #tf.keras.models.save('Neural_Network.h5')
    

    return model








Z1=np.random.rand(50,10,2)
Z2=np.zeros((50,1))

print(Z1)
#Z2=np.sum(np.all(Z1)**2)

#'''
print(np.shape(Z1))
#print(Z1)
for i in range(len(Z2)):
    Z2[i]=0
    for j in range(len(Z1[i])):
        Z2[i]+=((Z1[i][j][0]**2+Z1[i][j][1]**2)**0.5)/10
    
#'''


#plt.scatter(Z1,Z2,s=5)
#plt.show()


model=Deep_neural_net(Z1,Z2)

Z1=np.random.rand(20,10,2)
Z2=np.zeros((20,1))

#Z2=np.sum(np.all(Z1)**2)

#'''
print(np.shape(Z1))
#print(Z1)
for i in range(len(Z2)):
    Z2[i]=0
    for j in range(len(Z1[i])):
        Z2[i]+=((Z1[i][j][0]**2+Z1[i][j][1]**2)**0.5)/10
#'''

#model.load_model('Neural_Network.h5')

Answer=model.predict(Z1)

score=model.evaluate(Z1,Z2)

plt.plot(Z2)
plt.plot(Answer)
print(score)
#plt.scatter(Z1,Answer,s=10)
#plt.scatter(Z1,Z2,s=5)

plt.show()

#print(model.evaulate())


#model.output_shape
#model.get_weights()
#model.save('models.h5')