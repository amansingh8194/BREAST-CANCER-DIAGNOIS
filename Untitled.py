#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split


# In[2]:


#Data Collection & Processing
# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[3]:


print(breast_cancer_dataset)


# In[4]:


# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)


# In[5]:


#print the first 5 rows of the dataframe
data_frame.head()


# In[6]:


# adding the target column to the data frame
data_frame['label'] = breast_cancer_dataset.target


# In[7]:


# print last 5 rows of the dataframe
data_frame.tail()


# In[8]:


# number of rows and columns in the dataset
data_frame.shape


# In[9]:


# getting some information about the data
data_frame.info()


# In[10]:


# checking for missing values
data_frame.isnull().sum()


# In[11]:


# statistical measures about the data
data_frame.describe()


# In[12]:


# checking the distribution of Target Variable
data_frame['label'].value_counts()


# In[13]:


data_frame.groupby('label').mean()


# In[14]:


#Separating the features and target
X = data_frame.drop(columns='label' , axis = 1)
Y = data_frame['label']


# In[15]:


print(X)


# In[16]:


print(Y)


# In[17]:


#Splitting the data into training data & Testing data
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 2)


# In[18]:


print(X.shape , X_train.shape , X_test.shape)


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# In[ ]:





# In[21]:


# importing tensorflow and Keras
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras


# In[22]:


# setting up the layers of Neural Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (30,)),
    keras.layers.Dense(20, activation = 'relu'),
    keras.layers.Dense(2, activation = 'sigmoid')
])


# In[23]:


#compiling the Neural Network
model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[24]:


# training the Neural Network
history = model.fit(X_train_std, Y_train , validation_split = 0.1 , epochs = 10)


# In[25]:


loss , accuracy = model.evaluate(X_test_std , Y_test)
print(accuracy)


# In[26]:


print(X_test_std.shape)
print(X_test_std[0])


# In[27]:


Y_pred = model.predict(X_test_std)


# In[28]:


print(Y_pred.shape)
print(Y_pred[0])


# In[29]:


print(X_test_std)


# In[30]:


print(Y_pred)


# In[31]:


#  argmax function

my_list = [0.25, 0.56]

index_of_max_value = np.argmax(my_list)
print(my_list)
print(index_of_max_value)


# In[32]:


# converting the prediction probability to class labels

Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)


# In[35]:


input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0] == 0):
  print('The tumor is Malignant')

else:
  print('The tumor is Benign')


# In[ ]:




