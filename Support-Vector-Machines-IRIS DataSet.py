#!/usr/bin/env python
# coding: utf-8

# ___
# # Support Vector Machines Project 
# 
# 
# We will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

# 
# The three classes in the Iris dataset:
# 
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# 
# The four features of the Iris dataset:
# 
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm
# 

# In[101]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Get the data
# 
# **Use seaborn to get the iris data **

# In[102]:


iris=sns.load_dataset('iris')


# In[103]:


iris


# In[104]:


iris.info()


# In[105]:


iris.describe()


# In[106]:


# Setosa is the most separable. 
sns.pairplot(iris,hue='species',palette='Dark2')


# # Train Test Split
# 
# ** Split the data into a training set and a testing set.**

# In[107]:


from sklearn.model_selection import train_test_split


# In[108]:


X=iris.drop('species',axis=1)
y=iris['species']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.30)


# # Train a Model
# 
# Now its time to train a Support Vector Machine Classifier. 
# 
# **Call the SVC() model from sklearn and fit the model to the training data.**

# In[109]:


from sklearn.svm import SVC


# In[110]:


svc_model = SVC()


# In[111]:


svc_model.fit(X_train,y_train)


# ## Model Evaluation
# 
# **Now get predictions from the model and create a confusion matrix and a classification report.**

# In[112]:


predictions = svc_model.predict(X_test)


# In[113]:


from sklearn.metrics import classification_report,confusion_matrix


# In[114]:


print(confusion_matrix(y_test,predictions))


# In[115]:


print(classification_report(y_test,predictions))


# Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but just using GridSearch for sure.

# ## Gridsearch Practice
# 
# ** Import GridsearchCV from sklearn Learn.**

# In[116]:


from sklearn.model_selection import GridSearchCV


# **Create a dictionary called param_grid and fill out some parameters for C and gamma.**

# In[117]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 


# ** Create a GridSearchCV object and fit it to the training data.**

# In[118]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# ** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**

# In[119]:


grid_predictions = grid.predict(X_test)


# In[120]:


print(confusion_matrix(y_test,grid_predictions))


# In[121]:


print(classification_report(y_test,grid_predictions))


# as you can see the results are the same as each other
