#!/usr/bin/env python
# coding: utf-8

# In[8]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[9]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[10]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


iris = sns.load_dataset('iris')


# In[18]:


iris.head(3)


# In[31]:


sns.pairplot(iris,hue='species',palette='Dark2')


# In[49]:


sns.jointplot(x='sepal_width',y='sepal_length',kind='kde',data=iris,color='r',ratio=6)


# In[38]:


setosa=iris[iris['species']=='setosa']


# In[52]:


sns.jointplot(x=setosa['sepal_width'],y=setosa['sepal_length'],kind='kde',data=setosa,cmap='plasma',shade=True,ratio=10,color='g')


# In[54]:


from sklearn.model_selection import train_test_split
X=iris.drop('species',axis=1)
y=iris['species']


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)


# In[57]:


from sklearn.svm import SVC


# In[58]:


ml=SVC()


# In[59]:


ml.fit(X_train,y_train)


# In[60]:


predictions = ml.predict(X_test)


# In[61]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[62]:


print(classification_report(y_test,predictions))


# In[63]:


from sklearn.model_selection import GridSearchCV


# In[73]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf']} 


# In[74]:


ml2=GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[75]:


ml2.fit(X_train,y_train)


# In[76]:


ml2.best_params_


# In[77]:


ml2.best_estimator_


# In[78]:


grid_predictions = ml2.predict(X_test)


# In[79]:


print(confusion_matrix(y_test,grid_predictions))


# In[80]:


print(classification_report(y_test,grid_predictions))


# In[ ]:




