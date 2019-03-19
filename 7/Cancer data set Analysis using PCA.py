#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[4]:


cancer = load_breast_cancer()
cancer.keys()


# In[5]:


print(cancer['DESCR'])


# In[6]:


df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[7]:


df.head()


# In[8]:


from sklearn.preprocessing import StandardScaler


# In[9]:


scaler = StandardScaler()
scaler.fit(df)


# In[10]:


scaled_data = scaler.transform(df)


# In[11]:


from sklearn.decomposition import PCA


# In[12]:


pca = PCA(n_components=2)


# In[13]:


pca.fit(scaled_data)


# In[14]:


x_pca = pca.transform(scaled_data)


# In[15]:


scaled_data.shape


# In[16]:


x_pca.shape


# In[22]:


sns.set_style('darkgrid')
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma',alpha=0.8)
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[23]:


pca.components_


# In[24]:


df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])


# In[25]:


plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)


# In[ ]:




