
# coding: utf-8

# In[2]:


import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import datetime as dt


# In[3]:


flag_data_load = pd.read_csv("E:\data set\Flag\\flag_data.txt", delimiter = ',')
#flag_data_load.iloc[:,:]
#  list(flag_data_load) == gives the attribute list
#df = flag_data_load
#df = df.drop('name', 1)
#df


# In[4]:


# preparing input
X = flag_data_load
col_del = ['name', 'religion', 'toplest', 'botright', 'mainhue']
X = X.drop(col_del , 1)
#X = np.c_[np.ones(flag_data_load.shape[0]), X]
#X = np.array(X)
#X.shape

#X = X.astype(np.float)


# In[5]:


#preparing output
Y = flag_data_load.iloc[:,6]


# In[6]:


# preparing weights

W = np.random.randn(X.shape[1])*0.01
#W.shape == (29,)
#Y_cap = np.dot(X,W)


# In[7]:


# gradient error 
def propagate(W,X,Y):
    m = X.shape[0]
    Y_cap = np.dot(X,W)
    
    cost = (1/2*m)*np.sum(np.square(Y_cap - Y))
    grad = (1/m)*np.dot(X.T, Y_cap - Y)
    return grad, cost


# In[34]:


def optimize(W, X, Y, num_iterations, learning_rate, print_cost = False):
    #cos = np.zeros((1000,))   
    for i in range (num_iterations):
        grads, cost = propagate(W, X, Y)
        W = W - learning_rate*grads 
        #cos[i,] = cost
        i = i+1
    return W 


# In[35]:


def predict(W,X):
    m = Y.shape[0]
    Y_predict = np.zeros((m,))
    Y_cap = np.dot(X,W)
    
    for i in range(Y_predict.shape[0]):
        if Y_cap[i,] < 0.5:
            Y_predict[i,] = 0
        elif Y_cap[i,] >= 0.5 and Y_cap[i,] < 1.5 :
            Y_predict[i,] = 1
        elif Y_cap[i,] >= 1.5 and Y_cap[i,] < 2.5 :
            Y_predict[i,] = 2
        elif Y_cap[i,] >= 2.5 and Y_cap[i,] < 3.5 :
            Y_predict[i,] = 3
        elif Y_cap[i,] >= 3.5 and Y_cap[i,] < 4.5 :
            Y_predict[i,] = 4
        elif Y_cap[i,] >= 4.5 and Y_cap[i,] < 5.5 :
            Y_predict[i,] = 5
        elif Y_cap[i,] >= 5.5 and Y_cap[i,] < 6.5 :
            Y_predict[i,] = 6
        else :
            Y_predict[i,] = 7

    return Y_predict
    


# In[36]:


def model(W, X, Y, num_iterations  = 1000, learning_rate = 0.5, print_cost = False):
    W = optimize(W, X, Y, num_iterations, learning_rate, print_cost = False)
    m = Y.shape[0]
    Y_predict = np.zeros((m,))
    Y_predict = predict(W,X)
    
    # finding acccuracy
    
    Bcc = Y_predict - Y
    
    Acc = (Bcc == 0).sum()
    
    Acc_percentage = (Acc/Y.shape[0])*100

    return Acc_percentage, Y_predict, Acc
    
         
    

    
    


# In[37]:


Acc_percentage, Y_predict, Acc = model(W, X, Y, num_iterations  = 1000, learning_rate = 0.1, print_cost = False)
Acc_percentage

