
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

get_ipython().magic(u'matplotlib inline')


# In[29]:


files = ['./btc.csv','./ltc.csv','eth.csv']
i=0
data = []
for f_name in files:
    d = []
    with open(f_name) as f:
        f.readline()
        for line in f:
            content = line.split(',')
            if content[0]>='2015-08-10':
                d.append(float(content[4]))
    data.append(d)


# In[31]:


prediction_days = 30
df_train = [data[i][:len(data[i])-prediction_days] for i in range(3)]
df_test = [data[i][len(data[i])-prediction_days:] for i in range(3)]


# In[35]:


training_set = df_train
training_set = [np.reshape(training_set[i], (len(training_set[i]), 1)) for i in range(3)]

sc = [MinMaxScaler() for i in range(3)]

training_set = [sc[i].fit_transform(training_set[i]) for i in range(3)]
X_train = [training_set[i][0:len(training_set[i])-1] for i in range(3)]
y_train = [training_set[i][1:len(training_set[i])] for i in range(3)]
X_train = [np.reshape(X_train[i], (len(X_train[i]), 1, 1)) for i in range(3)]


# In[36]:


regressor = [Sequential() for i in range(3)]
for i in range(3):
    regressor[i].add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
    regressor[i].add(Dense(units = 1))
    regressor[i].compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor[i].fit(X_train[i], y_train[i], batch_size = 5, epochs = 100)


# In[37]:


test_set = df_test
test_set = [np.reshape(test_set[i], (len(test_set[i]), 1)) for i in range(3)]
test_set = [sc[i].transform(test_set[i]) for i in range(3)]
test_set = [np.reshape(test_set[i], (len(test_set[i]), 1, 1)) for i in range(3)]
y_pred = [regressor[i].predict(test_set[i]) for i in range(3)]
y_pred = [sc[i].inverse_transform(y_pred[i])for i in range(3)]


# In[54]:


for i in range(3):
    plt.figure(figsize=(10,10), dpi=70, facecolor='w', edgecolor='k')
    ax = plt.gca()  
    plt.plot(df_test[i], color = 'red', label = 'real c'+str(i+1)+' price')
    plt.plot(y_pred[i], color = 'blue', label = 'predicted c'+str(i+1)+' Price')
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('c'+str(i+1)+' price in USD', fontsize=10)
    plt.legend(loc=2, prop={'size': 10})
    plt.show()


# In[ ]:




