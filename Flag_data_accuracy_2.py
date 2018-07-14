
# coding: utf-8

# In[137]:


import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import datetime as dt


# In[138]:



flag_data_load = pd.read_csv("E:\data set\Flag\\flag_data.txt", delimiter = ',')
#flag_data_load.iloc[:,:]
#  list(flag_data_load) == gives the attribute list
#df = flag_data_load
#df = df.drop('name', 1)


# In[139]:



X = flag_data_load
col_del = ['name', 'religion', 'toplest', 'botright', 'mainhue']
X = X.drop(col_del , 1)
#X = np.c_[np.ones(flag_data_load.shape[0]), X]
#X = np.array(X)
#X.shape

#X = X.astype(np.float)
#X.shape == 194.25
    


# In[140]:


#preparing output

Y = flag_data_load.iloc[:,6]
b = np.zeros((flag_data_load.shape[0], 8 ))
b[np.arange(flag_data_load.shape[0]), Y] = 1
Y = b

    
#Y.shape == 194,8


# In[141]:


# preparing weights
def initialize():
    W = np.random.rand(X.shape[1],Y.shape[1] )
    b = np.zeros((1, Y.shape[1]))
#W.shape == (29,)
#Y_cap = np.dot(X,W)
#W.shape == 25,8
#b = 1,8
    return W,b


# In[142]:


def sigmoid(X,W, b):

    z = np.dot(X,W) + b
    Y_cap = 1/(1 + np.exp(-z))
    
    return Y_cap
    


# In[143]:


# gradient error 
def propagate(W,X,Y, b):
    m = X.shape[0]
    Y_cap  = np.sigmoid(X,W, b)
    cost = (-1/(2*m))*np.sum(np.dot(Y.T, np.log(Y_cap))  + np.dot((1-Y).T, np.log(1-Y_cap))) 
    #grad = (1/m)*np.dot(X.T, Y_cap - Y)
    return cost


# In[144]:


def optimize(W,b, X , Y , num_iterations, learning_rate, print_cost = False):
    m = X.shape[0]
   
    
    for i in range(num_iterations):
        Y_cap = sigmoid(X,W, b)
        dz = Y_cap  - Y
        dw = (1/m)*np.dot(X.T, dz)
        db = (1/m)*np.sum(dz, axis = 0)
        W  = W - learning_rate*dw
        b = b - learning_rate*db
        
    return W,b
    
    #cos = np.zeros((1000,))   


# In[145]:


def predict(Y_cap):
    m = Y.shape[0]
    Y_predict = np.zeros((m,1))
    
    Y_predict = np.argmax(Y_cap, axis=1)            

    return Y_predict
    


# In[148]:


def model(W,b,  X, Y , num_iterations , learning_rate , print_cost = False):
    m = Y.shape[0]
    W, b = optimize(W,b, X, Y, num_iterations = 1000, learning_rate = 0.01, print_cost = False)
    Y_cap  = sigmoid(X,W, b)
    
    Y_predict = predict(Y_cap)
    
    # finding acccuracy
    
    Bcc = Y_predict - flag_data_load.iloc[:,6]
    
    Acc = (Bcc == 0).sum()
    
    Acc_percentage = (Acc/Y.shape[0])*100

    return Acc_percentage, Y_predict, Acc
    
         
    

    
    


# In[150]:


W,b = initialize()
Acc_percentage, Y_predict, Acc = model(W,b,  X , Y , num_iterations  = 1000, learning_rate = 0.01, print_cost = False)
Acc_percentage

