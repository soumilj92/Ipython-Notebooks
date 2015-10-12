
# coding: utf-8

# In[34]:

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
import math


# In[35]:

get_ipython().system(u'ls')


# In[65]:

test=pd.read_csv('test.csv',header=None)


# In[37]:

test.head(2)


# In[38]:

test.shape


# In[39]:

train=pd.read_csv('train.csv',header=None)
train.head(3)


# In[40]:

train.shape


# In[41]:

train_y=pd.read_csv('labels.csv',header=None)
train_y.head(2)


# In[42]:

train[40]=0
train[40]=train_y
train.head(2)


##### So now we have the prepared train dataset where the 41st column carries the value of the dependent variable

# In[43]:

t=train
train=pd.read_csv('train.csv',header=None)


### Bivariate analysis

#### Pearson correlation coefficient 

##### Pearson correlation coefficient is the standard correlation coefficient between 2 variables that is defined as the ratio of  covariance between the 2 variables and the square root of the product of variances of the 2 variables 

# In[44]:

from scipy.stats.stats import pearsonr


# In[45]:

pearsonr(train[0],train[1])[0]


##### the first value is the correlation coefficient while the second one is the p value of the null hypothesis so clearly lesser the p value of the hypothesis better the correlation coefficient accuracy 

# In[46]:

train.head()


##### We will calculate the pearson correlation coefficient for each of the bivariable combination of the 40 variables 

# In[51]:

temp=np.zeros((40,40))
for i in np.arange(0,40,1):
    for j in np.arange(0,40,1):
        temp[i][j]=pearsonr(train[i],train[j])[0]


# In[52]:

C=DataFrame(temp)
C1=C


# In[53]:

C1.head(1)


##### We want to see the correlation amongst the different variables , so we eliminate those variables with very low correlation coefficient 

# In[54]:

C[(C>-0.35) & (C<0.35)]=0


##### Now count the number of non zero values for each of the variable 

##### store it in a numpy array for further analysis

# In[55]:

S=np.array([])
for i in np.arange(0,40,1):
    S=np.append(S,np.count_nonzero(C[i]))


# In[56]:

S


##### But this is not true in reality , 1 value is due to the fact that the variable correlation with itself is always 1 so make those 1's also 0

# In[57]:

for i in np.arange(0,40,1):
    C[i][i]=0


##### Now run the procedure for S again 

# In[58]:

S=np.array([])
for i in np.arange(0,40,1):
    S=np.append(S,np.count_nonzero(C[i]))


# In[59]:

S


##### So we have now got the required correlation coefficients which could be significant in our final model building

#### KMeans Clustering Analysis 

# In[60]:

get_ipython().magic(u'pylab inline')


# In[61]:

from sklearn.cluster import KMeans


# In[62]:

Estimator=KMeans(init='k-means++',n_clusters=2,n_init=10)
Estimator


# In[63]:

model=Estimator.fit(train)
model


# In[67]:

test.head(2)


# In[68]:

result=np.array([])
for i in np.arange(0,9000,1):
    result=np.append(result,model.predict(test.ix[i]))


# In[69]:

len(result)


# In[70]:

submission=DataFrame({'Id':np.arange(1,9001,1),'Solution':result})


# In[71]:

submission.head()


# In[75]:

submission.to_csv('submit1_KMeans.csv',index=False)


##### We have to also convert the solution column to int type from float else a 0 score is obtained

# In[82]:

submission.info()


# In[91]:

submission['Solution']=submission['Solution'].astype(int)


# In[92]:

submission.info()


# In[93]:

submission['Solution'].value_counts()


# In[94]:

submission.to_csv('submit1_KMeans.csv',index=False)


## Score : .49814

## Random forests 

##### Random forests are an ensemble learning method for classification (and regression) that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes output by individual trees. The term came from random decision forests that was first proposed by Tin Kam Ho of Bell Labs in 1995. The method combines Breiman's "bagging" idea and the random selection of features, introduced in order to construct a collection of decision trees with controlled variance.

# In[99]:

from sklearn import ensemble


# In[100]:

clf = ensemble.RandomForestClassifier(n_estimators = 500, n_jobs = -1)


# In[103]:

train.shape


# In[106]:

train_y=pd.read_csv('labels.csv',header=None)


# In[108]:

clf.fit(train,train_y)


# In[109]:

test_y=clf.predict(test)


# In[110]:

type(test_y)


# In[111]:

test_y


# In[115]:

submission['Solution']=Series(test_y)


# In[116]:

submission.head(2)


# In[117]:

submission.shape


# In[118]:

submission.to_csv('random_forests.csv',index=False)


## Score :0.87109

##### As the data is small , we can try random forest with a greater number of estimators 

# In[127]:

clf = ensemble.RandomForestClassifier(n_estimators = 1000, n_jobs = -1)
clf.fit(train,train_y)
test_y=clf.predict(test)
submission.to_csv('random_forests1000.csv',index=False)
submission.head(2)


## Score :0.87444

# In[126]:

clf = ensemble.RandomForestClassifier(n_estimators = 10000, n_jobs = -1)
clf.fit(train,train_y)
test_y=clf.predict(test)
submission.to_csv('random_forests10000.csv',index=False)
submission.head(2)


### Removing some of the non correlated variables 

# In[128]:

from scipy.stats.stats import pearsonr


# In[167]:

corrr=np.array([])
p_value=np.array([])
for i in np.arange(0,40,1):
    corrr=np.append(corrr,pearsonr(train[i],train_y[0])[0])
    p_value=np.append(p_value,pearsonr(train[i],train_y[0])[1])


# In[168]:

print len(corrr),len(p_value)


# In[169]:

corr=DataFrame({'corelation':corrr,'p_value':p_value})


# In[170]:

corr.head(2)


# In[171]:

corr['p_value'].describe()


## Logistic Regression 

#### Missing Value Imputation 

# In[95]:

train.info()


##### As there is no missing value and all are float type variables , we dont need any missing value imputation or dummy variable creation as there is no characteristic variable 

# In[207]:

from sklearn import linear_model


# In[177]:

t[40].value_counts()


# In[178]:

y=t[40]
type(y)


# In[180]:

lr = linear_model.LogisticRegression()
lr.fit(train, y)


# In[181]:

test.head(2)


# In[182]:

predicted_probs = lr.predict_proba(test)


# In[183]:

predicted_probs


# In[184]:

predicted_probs.shape


# In[185]:

p=DataFrame(predicted_probs)


# In[186]:

p.shape


# In[187]:

p.head(2)


# In[188]:

p.describe()


# In[189]:

get_ipython().magic(u'pinfo lr.predict_proba')


# In[190]:

p1=p[0]
p2=p[1]


# In[192]:

p1[p1<0.5]=0
p1[p1==0.5]=0
p1[p1>0.5]=1
p2[p2<0.5]=0
p2[p2==0.5]=0
p2[p2>0.5]=1


# In[198]:

p1=np.array(p1)
p2=np.array(p2)


# In[195]:

p1


# In[199]:

print sum(p1),sum(p2)


# In[202]:

submission['Solution']=p2


# In[203]:

submission.head(2)


# In[204]:

submission.to_csv('logistic_regression.csv',index=False)


# In[205]:

submission['Solution']=submission['Solution'].astype(int)


# In[206]:

submission.to_csv('logistic_regression.csv',index=False)


## Score :0.80179

## Univariate analysis

##### Here we see how the various independent variables are distributed 

# In[253]:

train.head(1)


# In[254]:

train[0].hist()


##### we will combine the test and the train data as we are computing the distributions of only the independent variables 

# In[255]:

pieces=[train,test]


# In[256]:

comb=pd.concat(pieces)


# In[258]:

comb.index


# In[259]:

comb.index=np.arange(0,10000,1)


# In[262]:

fig1=comb[0].hist()


# In[263]:

comb[0].describe()


# In[264]:

comb.boxplot(column=[0])


## Q-Q Plot 

##### In statistics, a Q–Q plot ("Q" stands for quantile) is a probability plot, which is a graphical method for comparing two probability distributions by plotting their quantiles against each other. If the two distributions being compared are similar, the points in the Q–Q plot will approximately lie on the line y = x. If the distributions are linearly related, the points in the Q–Q plot will approximately lie on a line, but not necessarily on the line y = x.

# In[266]:

import statsmodels.api as sm


# In[269]:

sm.qqplot(comb[1],line='45')


# In[275]:

os.getcwd()


# In[287]:

for i in np.arange(0,40,1):
    pieces1='histograms/histogram',format(i),'.jpg'
    hist=comb[i].hist()
    fig = hist.get_figure()
    fig.savefig(''.join(pieces1))
    fig.clear()


##### The above code saves all the histograms for the 40 variables 

# In[285]:

for i in np.arange(0,40,1):
    sm.qqplot(comb[i],line='45')


# In[291]:

sm.qqplot(comb[4],line='45')


## Principal component analysis

#### Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.The number of principal components is less than or equal to the number of original variables. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to (i.e., uncorrelated with) the preceding components. The principal components are orthogonal because they are the eigenvectors of the covariance matrix, which is symmetric. PCA is sensitive to the relative scaling of the original variables.

##### The main purposes of a principal component analysis are the analysis of data to identify patterns and finding patterns to reduce the dimensions of the dataset with minimal loss of information.

# In[292]:

from matplotlib.mlab import PCA as mlabPCA


# In[295]:

mlab_pca = mlabPCA(train)
mlab_pca


# In[296]:

mlab_pca.Y


# In[298]:

mlab_pca.Y.shape


# In[299]:

PCAY=DataFrame(mlab_pca.Y)


# In[300]:

PCAY.head(2)


# In[301]:

train.head(2)


# In[309]:

from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_estimators = 1000, n_jobs = -1)
clf.fit(PCAY,train_y)
test_y=clf.predict(test)
submission['Solution']=test_y
submission.to_csv('PCA.csv',index=False)
submission.head(2)


# In[310]:

submission.info()


### score: 0.5182

##### The PCA analysis will be done separately in a more detailed manner to get a good score for the contest . I would still recommend that all  the analysis should be done independently to make urself familiar with all the algorithms 

# In[ ]:



