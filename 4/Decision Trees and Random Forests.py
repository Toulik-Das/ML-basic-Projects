#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df=pd.read_csv('loan_data.csv')


# In[7]:


df.head()


# In[8]:


df.info()


# In[12]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[13]:


df.describe()


# In[14]:


df.head(3)


# In[40]:



df[df['credit.policy']==1]['fico'].hist(bins=40,color='b',alpha=0.6,label='credit.policy=1',figsize=(10,6))
df[df['credit.policy']==0]['fico'].hist(bins=40,color='r',alpha=0.6,label='credit.policy=0',figsize=(10,6))
plt.legend()
plt.xlabel('FICO')


# In[39]:


df[df['not.fully.paid']==1]['fico'].hist(bins=40,color='b',alpha=0.6,label='not.fully.paid=1',figsize=(10,6))
df[df['not.fully.paid']==0]['fico'].hist(bins=40,color='r',alpha=0.6,label='not.fully.paid=0',figsize=(10,6))
plt.legend()
plt.xlabel('FICO')


# In[54]:


plt.figure(figsize=(10,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=df,palette='Set1')


# In[57]:


sns.pairplot(df,palette='Set2')


# In[72]:


plt.figure(figsize=(10,10))
sns.jointplot(x='fico',y='int.rate',data=df,color='purple',ratio=6,height=7)


# In[86]:


plt.figure(figsize=(12,7))
sns.lmplot(x='fico',y='int.rate',data=df,hue='credit.policy',col='not.fully.paid',palette='Set2')


# In[112]:


cat_feats = ['purpose']
final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)


# In[113]:


final_data.info()


# In[114]:


from sklearn.model_selection import train_test_split


# In[120]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']


# In[125]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[126]:


from sklearn.tree import DecisionTreeClassifier


# In[127]:


dtree = DecisionTreeClassifier()


# In[128]:


dtree.fit(X_train,y_train)


# In[129]:


predictions = dtree.predict(X_test)


# In[130]:


from sklearn.metrics import classification_report,confusion_matrix


# In[131]:


print(classification_report(y_test,predictions))


# In[132]:


print(confusion_matrix(y_test,predictions))


# In[133]:


list(df.columns)


# In[143]:


#Random Forests


# In[144]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[145]:


rfc_pred = rfc.predict(X_test)


# In[146]:


confusion_matrix(y_test,rfc_pred)


# In[148]:


print(classification_report(y_test,rfc_pred))


# In[ ]:




