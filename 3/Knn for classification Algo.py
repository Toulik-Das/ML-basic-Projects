#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df=pd.read_csv("KNN_Project_Data",index_col=0)


# In[6]:


df.head()


# In[60]:


sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')


# In[61]:


from sklearn.preprocessing import StandardScaler


# In[62]:


sc=StandardScaler()


# In[63]:


X=df.drop('TARGET CLASS',axis=1)


# In[64]:


sc.fit(X)


# In[65]:


scaled_data = sc.transform(X)


# In[67]:


df_feat = pd.DataFrame(scaled_data,columns=df.columns[:-1])
df_feat.head(3)


# In[34]:


from sklearn.model_selection import train_test_split


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(df_feat,df['TARGET CLASS'],
                                                    test_size=0.3,random_state=101)


# In[69]:


from sklearn.neighbors import KNeighborsClassifier


# In[70]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[78]:


knn.fit(X_train,y_train)


# In[72]:


prob = knn.predict(X_test)


# In[73]:


from sklearn.metrics import classification_report,confusion_matrix


# In[74]:


print(confusion_matrix(y_test,prob))


# In[75]:


print(classification_report(y_test,prob))


# In[85]:


error_rate = []

for i in range(1,100):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[95]:


plt.figure(figsize=(10,10))
plt.plot(range(1,100),error_rate,color='g', linestyle='dashed', marker='o',
         markerfacecolor='b', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[87]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[91]:


knn = KNeighborsClassifier(n_neighbors=37)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=37')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




