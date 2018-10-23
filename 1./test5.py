import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import numpy as np


#X = [[0., 1.], [1., 1.],[1., 0.]]
#y = [0, 1, 0]
X=[]
Y=[]


for i in range(2): 
    #np.append(X,np.loadtxt('../Simulations/Genes%i'%i))
    #np.append(Y,np.loadtxt('../Simulations/Power%i'%i))
    X.append(np.ndarray.tolist(np.loadtxt('../Simulations/Genes%i'%i)))
    Y.append(np.ndarray.tolist(np.loadtxt('../Simulations/Power%i'%i)))
    
    #np.concatenate(X,)
    #np.concatenate(Y,np.loadtxt('../Simulations/Power%i'%i))



clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(25, 2), random_state=1)




clf.fit(X, Y)                         
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(25, 3),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)


X=[]
Y=[]

for i in range(2,4): 
    #np.append(X,np.loadtxt('../Simulations/Genes%i'%i))
    #np.append(Y,np.loadtxt('../Simulations/Power%i'%i))
    X.append(np.ndarray.tolist(np.loadtxt('../Simulations/Genes%i'%i)))
    Y.append(np.ndarray.tolist(np.loadtxt('../Simulations/Power%i'%i)))

print(clf.predict(X))

print(Y)


#print([coef.shape for coef in clf.coefs_])

#print(clf.predict_proba([[2., 2.], [1., 2.]]))


