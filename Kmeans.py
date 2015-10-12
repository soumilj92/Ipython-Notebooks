
# coding: utf-8

# In[166]:

import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D #for 3d plots
from pandas import DataFrame,Series
import pandas as pd
import numpy as np
import os # for directory operations 
from time import time # to measure the time of code compilation


##### time function from time module is very sweet as it extracts the current time from the sys which can be stored in a variable

# In[167]:

time()


# In[168]:

get_ipython().magic(u'pylab inline')


# In[169]:

iris=datasets.load_iris()


# In[170]:

type(iris)


# In[171]:

#bunch kind of a dataset


##### K-nearest neighbors classifier - In this method of variable classification basically we divide the data points into K number of groups each representing its own set of properties

# In[172]:

#create a variable classifier 
from sklearn import neighbors
knn=neighbors.KNeighborsClassifier()


# In[173]:

#now we will fit our dataset into the classifier model
knn.fit(iris.data,iris.target)


# In[174]:

p=knn.predict([[0.1,0.2,0.3,0.4]])
# we give all the 4 features for which we want to predict the value of the
#dependent variable


# In[175]:

p


# In[176]:

type(iris.data)


# In[177]:

iris.target


# In[178]:

len(iris.target)


# In[179]:

Y=Series(iris.target)


# In[180]:

Y.value_counts()


##### Clearly there are 3 groups of dependent variable values .Now we will try to depict these values graphically  

# In[181]:

iris.data.shape


##### For each of the dependent variable in Y , we have 4 independent variables - we will use 2 of the independent variables and try to plot the points on a plot

# In[182]:

X=DataFrame(iris.data[:,0:2])


##### Thus we have extracted the first 2 columns of the dataset 

# In[183]:

v1=X[0]
v2=X[1]


##### The 2 features we took in the above case are the sepal width and the sepal length

# In[184]:

plt.scatter(v1,v2)
plt.ylabel('Sepal Width')
plt.xlabel('Sepal Length')
plt.title('Kmeans_clustering analysis')


##### The above graph clearly depicts that there could be some classification done using the K-means clustering as there is a big gap between the points on the north and on the south 

#### 1.The only parameter in a K-means clustering is the number of clusters that is k

#### 2.The geometry(Metric used) for K-means clustering is the distance between points 

# In[185]:

from sklearn.cluster import KMeans


##### Lets define the estimator first 

# In[186]:

from sklearn.decomposition import PCA


##### PCA is Principal component analysis . Linear dimensionality reduction using singular value Decomposition of the data and keeping only the most significant singular vectors to project the data to a lower dimentional space 

##### There is no need to reduce the data in this case so PCA wont be used here in this case 

# In[187]:

#n_clusters here is the number of clusters we want to make 
#as there are 3 unique values of the dependent variables
#lets try to visualize the results using 3 clusters only
Estimator=KMeans(init='k-means++',n_clusters=3,n_init=10)
Estimator


##### Now we will fit the data iris into the estimator 

# In[188]:

C=Estimator.fit(X)
C


##### Now using the above cluster analysis we can predict the cluster for other data points 

# In[189]:

C.predict([.2,.3])


# In[190]:

C.predict([7.2,3.2])


#### How to find the value of K , if we dont really know anything about the data but we still want to do the KMeans clustering analysis

# In[191]:

from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist


# In[192]:

K=range(1,11)
K


# In[193]:

iris.data.shape


# In[194]:

X=iris.data


##### Now we will apply kmeans for each value of k from 1 to 10 

# In[195]:

KM=[kmeans(X,k) for k in K]
print type(KM),len(KM)


# In[196]:

KM_df=DataFrame(KM)
print KM_df.head(1)


# In[197]:

print KM_df.tail(1)


# In[198]:

KM_df.shape


# In[199]:

KM_v1=KM_df[0]
print type(KM_v1)


# In[200]:

KM_v1[0]


# In[201]:

KM_v1[0][0]


# In[202]:

print type(KM_v1[0][0]),len(KM_v1[0][0])


# In[203]:

for i in range(0,10):
    print len(KM_v1[i]),len(KM_v1[i][0])


##### This is because in case of 1 cluster there is only 1 set of possible values for the 4 variables and in 2 clusters there are 2 centroids so 2 4 cross 1 arrays representing the 2 centroids 

##### Now we will calculate the centroid for each of the cluster . These will be nothing but the first column of the above defined KM dataframe 

# In[204]:

Centroids=KM_df[0]


##### Now we will calculate the total euclidian that is nothing but the geaometric modulus distance of the points form the centroid 

# In[205]:

dist=[cdist(X,cent,'euclidean') for cent in Centroids]
dist_df=DataFrame(dist)


# In[206]:

dist_df.shape


# In[207]:

dist_df.head(1)


##### As there is only 1 column in the dataframe we can convert it into a series and then analyse it 

# In[208]:

dist_series=Series(dist_df[0])


##### Now we will try to find the distance of the centroid in the cluster in case of K=1 and the first point out of the 150 points in the dataset 

# In[209]:

dist_series[0][0]


##### Now we all know that each of the 150 point in the original dataset has to be given a particular cluster . We will calculate the minimum distance of a points from one of the centroid 

# In[210]:

dist_series[1][0]


# In[211]:

min(dist_series[1][0])


##### So for k=2, the first point will be in the first cluster clearly as its distance from the first cluster's centroid is much lesser 

##### so for each of 150 points in the original dataset and for each of the value of K , we will calculate the minimum distance of the point form a centroid 

##### we will make a (150,10) dataframe and store these minimum distance values in them 

# In[212]:

dist_series


# In[213]:

min(dist_series[4][149])


# In[214]:

temp=np.array([])
for i in np.arange(0,150,1):
    for j in np.arange(0,10,1):
        temp=np.append(temp,min(dist_series[j][i]))


# In[215]:

temp


# In[216]:

print len(temp),temp[0],type(temp)


##### Now temp contains all the values in the order of clusterwise

##### We will now store these values in a 150  cross 10 dataframe

# In[217]:

DF=DataFrame(temp.reshape((150,10)))


##### So now we have the dataframe that contains all the minimum distance values in it 

##### Now lets find the average distance for each value of k

# In[218]:

sum_dist=np.array([])
for i in np.arange(10):
    sum_dist=np.append(sum_dist,sum(DF[:][i]))


# In[219]:

sum_dist


##### So for each cluster we know the average distance of the points from the centroids 

##### Now to know the optimum value of k , we will plot the elbow curve 

# In[220]:

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(K,sum_dist,'b*-')
#Here b means a blue line in the graph 
# * is to make * as the points on the graph
# dash is the dashed line that should be made 
plt.grid(True)
#just to make a background grid
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for K-Means clustering')  


##### Clearly as we increase the number of clusters, the average distance keeps on decreasing .The optimum value is clearly 3 because after that value thereis not a much decrease and thus we dont need to have the 4th cluster  

##### Lets now finally try to plot the clusters on a graph for 2 variables 

# In[221]:

from sklearn.cluster import KMeans
Estimator=KMeans(init='k-means++',n_clusters=3,n_init=10)
Estimator


# In[222]:

C=Estimator.fit(X)
C


# In[223]:

type(X)


# In[228]:

Y=C.predict(X)
type(Y)


# In[229]:

Y


# In[235]:

X=DataFrame(X)
X_t=X
X.columns


# In[236]:

#We make an extra column to add the Y values (dependent variable value)
X_t[4]=0
X_t[4]=Y


# In[237]:

cols=['b','g','r']


# In[240]:

fig=plt.figure()
ax=fig.add_subplot(111)
plt.grid(True)
plt.ylabel('Sepal Width')
plt.xlabel('Sepal Length')
plt.title('Kmeans_clustering analysis')  
for i in np.arange(0,150,1):
        ax.scatter(X_t[0][i],X_t[1][i],color=cols[X_t[4][i]])


##### Thus we see the 3 clusters clearly separated in the plot 

# In[ ]:



