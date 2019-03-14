#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('advertising.csv')


# In[6]:


df.head()


# In[4]:


df.info()


# In[7]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[9]:


sns.heatmap(df.isnull(),cbar=True,cmap='viridis')


# In[11]:


df.isnull()


# In[20]:


df['Age'].hist(bins=40,color='darkred',alpha=0.7)


# In[47]:


plt.hist(df['Age'],bins=40)


# In[18]:


df.info()


# In[43]:


sns.distplot(df['Age'],kde=True,color='darkred',bins=50)


# In[30]:


g=sns.JointGrid(x="Age", y="Area Income", data=df)
g = g.plot(sns.regplot, sns.distplot)


# In[32]:


g=sns.JointGrid(x="Daily Time Spent on Site", y="Daily Internet Usage", data=df)
g = g.plot(sns.regplot, sns.distplot)


# In[34]:


sns.jointplot(data=df,x='Daily Time Spent on Site',y='Daily Internet Usage',kind='scatter',height=6,ratio=5,color='g')


# In[8]:


sns.jointplot(x="Age", y="Area Income",data=df,kind='kde',color='red')


# In[7]:


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=df,kind='kde')


# In[41]:


sns.pairplot(df,palette='coolwarm')


# In[49]:


df.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1,inplace=True)
df.head()


# In[48]:


from sklearn.model_selection import train_test_split


# In[50]:


X=df.drop('Clicked on Ad',axis=1)
y=df['Clicked on Ad']


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[54]:


from sklearn.linear_model import LogisticRegression


# In[55]:


lm=LogisticRegression()


# In[58]:


lm.fit(X_train,y_train)


# In[60]:


predictions = lm.predict(X_test)


# In[61]:


from sklearn.metrics import classification_report


# In[62]:


print(classification_report(y_test,predictions))


# In[63]:


from sklearn import metrics


# In[64]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[65]:


print(lm.intercept_)
print(lm.coef_)
X.columns


# In[67]:


plt.scatter(y_test,predictions,color='b')


# In[68]:


sns.distplot((y_test-predictions),bins=50,color='b');


# In[ ]:




