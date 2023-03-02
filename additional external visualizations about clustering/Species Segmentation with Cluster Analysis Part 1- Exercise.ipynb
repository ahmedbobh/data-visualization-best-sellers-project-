#!/usr/bin/env python
# coding: utf-8

# # Species Segmentation with Cluster Analysis

# The Iris flower dataset is one of the most popular ones for machine learning. You can read a lot about it online and have probably already heard of it: https://en.wikipedia.org/wiki/Iris_flower_data_set
# 
# We didn't want to use it in the lectures, but believe that it would be very interesting for you to try it out (and maybe read about it on your own).
# 
# There are 4 features: sepal length, sepal width, petal length, and petal width.
# 
# Start by creating 2 clusters. Then standardize the data and try again. Does it make a difference?
# 
# Use the Elbow rule to determine how many clusters are there.
# 

# ## Import the relevant libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# ## Load the data

# Load data from the csv file: <i> 'iris_dataset.csv'</i>.

# In[3]:


data=pd.read_csv('iris-dataset.csv')
data.head()


# ## Plot the data

# For this exercise, try to cluster the iris flowers by the shape of their sepal. 
# 
# <i> Use the 'sepal_length' and 'sepal_width' variables.</i> 

# In[5]:


plt.scatter(data['sepal_length'],data['sepal_width'])


# # Clustering (unscaled data)

# Separate the original data into 2 clusters.

# In[6]:


kmeans=KMeans(2)
kmeans.fit(data)


# In[7]:


cluster=data.copy()
cluster['clusters']=kmeans.fit_predict(data)
cluster


# In[8]:


plt.scatter(cluster['sepal_length'],cluster['sepal_width'],c=cluster['clusters'],cmap='rainbow')


# # Standardize the variables

# Import and use the <i> method </i> function from sklearn to standardize the data. 

# In[9]:


from sklearn import preprocessing


# # Clustering (scaled data)

# In[10]:


data_scaled=preprocessing.scale(data)


# In[11]:


data_scaled


# In[ ]:





# In[ ]:





# ## Take Advantage of the Elbow Method

# ### WCSS

# In[12]:


wcss=[]
for i in range(1,5):
    kmeans=KMeans(i)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)
wcss


# ### The Elbow Method

# In[13]:


plt.plot(range(1,5),wcss)
plt.xlabel('number of clusters')
plt.ylabel('wcss')


# How many clusters are there?
