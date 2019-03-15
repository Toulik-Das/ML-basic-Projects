#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('College_Data',index_col=0)


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[16]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.scatterplot(x='Room.Board',y='Grad.Rate',hue='Private',data=df)


# In[17]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.scatterplot(x='Outstate',y='F.Undergrad',hue='Private',data=df)


# In[18]:


sns.set_style('whitegrid')
g = sns.FacetGrid(df,hue='Private',height=6,aspect=2,palette='Set2')
g = g.map(plt.hist, "Outstate",bins=30)


# In[21]:


sns.set_style('whitegrid')
g = sns.FacetGrid(df,hue='Private',height=6,aspect=2,palette='coolwarm')
g = g.map(plt.hist, "Grad.Rate",bins=30)


# In[10]:


df[df['Grad.Rate'] > 100]


# In[25]:


df['Grad.Rate']['Cazenovia College'] = 100


# In[26]:


df[df['Grad.Rate'] > 100]


# In[27]:


sns.set_style('whitegrid')
g = sns.FacetGrid(df,hue='Private',height=6,aspect=2,palette='coolwarm')
g = g.map(plt.hist, "Grad.Rate",bins=30)


# In[28]:


from sklearn.cluster import KMeans


# In[40]:


kmeans = KMeans(n_clusters=2)


# In[41]:


kmeans.fit(df.drop('Private',axis=1))


# In[42]:


kmeans.cluster_centers_


# In[43]:


kmeans.labels_


# In[44]:


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[45]:


df['Cluster'] = df['Private'].apply(converter)


# In[46]:


df.head()


# In[47]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))


# In[48]:


print(classification_report(df['Cluster'],kmeans.labels_))


# In[ ]:





# In[ ]:




