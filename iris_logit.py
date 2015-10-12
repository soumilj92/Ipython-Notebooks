
# coding: utf-8

# In[8]:

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import datasets
import statsmodels.api as sm
from pandas import DataFrame,Series


# In[2]:

get_ipython().magic(u'pylab inline')
iris=datasets.load_iris()


# In[3]:

print type(iris),type(iris.data),type(iris.target)


# In[4]:

iris.data.shape


##### Thus there are 150 observations each with 4 independent variables 

# In[5]:

iris.target.shape


##### 1 dependent variable 

# In[7]:

X=iris.data
Y=iris.target


# In[9]:

X[0]


# In[10]:

X=DataFrame(X)


# In[11]:

X.head(1)


# In[12]:

Y=DataFrame(Y)


# In[13]:

Y.head(1)


### Univariate analysis

# In[14]:

X.hist()


##### These histograms depicts the distribution of the 4 independent variables

##### We can do this analysis using just 1 variable also 

# In[15]:

X[0].hist()


##### we can also get the stats of that variable

# In[16]:

X[0].describe()


# In[22]:

X.boxplot(column=[0,1,2,3])


##### This is what is called a boxplot or a tail and a whisker diagram which gives the kind of distribution of each variable under study 

##### The red line in the boxplot gives the mean of that variable  The length of the box gives the spread of the variable  The minimum and maximum values are represented by the whisker ends The box endings are the lower and upper quantiles 

# In[38]:

X[4]=0
X.head(1)


# In[40]:

X[4]=Y


# In[41]:

X.boxplot(column=[0],by=[4])


# In[43]:

Y[0].value_counts()


### Performing the regression 

# In[44]:

from sklearn.linear_model import LogisticRegression


# In[45]:

get_ipython().set_next_input(u'logit=LogisticRegression');get_ipython().magic(u'pinfo LogisticRegression')


                
                
# In[91]:

logit=LogisticRegression(C=1e5)


# In[50]:

type(Y)


##### But we need a numpy array to get the logistic regression work 

# In[89]:

Y=iris.target
Y


# In[95]:

X=iris.data[:,:2]


##### We only will take the first 2 features of the X first

# In[96]:

logit.fit(X,Y)


##### By fitting the model we have created an instance of Neighbours classifier and fit the data

#### Plotting the decision boundary .

# In[97]:

x_min=X[:,0].min()-0.5


# In[98]:

x_max=X[:,0].max()+0.5


# In[99]:

y_min=X[:,1].min()-0.5


# In[100]:

y_max=X[:,1].max()+0.5


# In[103]:

#our step size is h=0.02
h=0.02


# In[104]:

xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))


# In[105]:

z=logit.predict(np.c_[xx.ravel(),yy.ravel()])


# In[107]:

z=z.reshape(xx.shape)


# In[116]:

plt.figure(1,figsize=(4,3))
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Paired)
plt.scatter(X[:,0],X[:,1],c=Y,edgecolors='k',cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()


# In[ ]:



