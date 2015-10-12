
# coding: utf-8

## Time Series Analysis  Kaggle - Bike Model 

# In[282]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[283]:

get_ipython().magic(u'pylab inline')


# In[2]:

os.getcwd()


# In[3]:

os.chdir('desktop/bike')


# In[4]:

get_ipython().system(u'ls')


# In[19]:

test=pd.read_csv('test.csv')


# In[20]:

test.head(1)


# In[21]:

type(test.datetime[0])


##### Lets convert the datetime column into a datetime format

# In[26]:

test=pd.read_csv('test.csv',parse_dates=['datetime'])


# In[27]:

test.head()


# In[28]:

type(test.datetime[0])


##### So now each element of the datetime column is a pandas timestamp

##### Now lets make datetime as the index of the dataframe

# In[30]:

test=pd.read_csv('test.csv',parse_dates=['datetime'],index_col='datetime')


# In[31]:

test.head(1)


# In[32]:

test.index


##### So we see that the index of the dataframe is a datetime index

# In[33]:

len(test.index)


# In[35]:

test.head(4)


# In[36]:

test_last=test.index[-1:]


# In[37]:

test_last


##### Lets make a time series from the first and the last datetime stamp of the above dataframe with an interval of an hour 

# In[44]:

test.index[0]


# In[46]:

test.index[len(test)-1]


# In[47]:

temp1=pd.date_range(test.index[0],test.index[len(test)-1],freq='H')


# In[48]:

temp1[:4]


# In[49]:

len(temp1)


# In[50]:

len(test)


##### So clearly there are a lot of time stamps missing in the dataframe test

##### Now similar to the test data make the train data 

# In[52]:

train=pd.read_csv('train.csv',parse_dates=['datetime'],index_col='datetime')


# In[54]:

train.head(3)


##### Clearly the sum of the casual and registered trips of bicycle should be equal to the count variable  .Lets check it once quickly

# In[55]:

l=0
for i in range(0,len(train),1):
        if ((train['casual'][i]+train['registered'][i])!=train['count'][i]):
            l=1
            break


# In[56]:

l


##### Therefore our assumption was correct 

##### Now lets try to study the various kind of variables first 

# In[57]:

train.describe()


# In[58]:

train['season'].value_counts()


##### This function gives the unique value count of a variable 

# In[61]:

train['holiday'].value_counts()


##### Holiday is a binary variable 

##### Clearly 1 represents a holiday and 0 is a non holiday .

# In[62]:

train['workingday'].value_counts()


##### Again working day is a binary variable 

# In[63]:

train['weather'].value_counts()


##### Again weather is a characteristic variable like seasons . There is only 1 count for weather=4 . So while creating the dummy variables , there is no need to make 3 dummy variables for weather . 2will be enough .

# In[65]:

train.info()


##### Clearly temp ,atemp and humidity are continuous variables 

##### Humidity is an integer variable 

# In[68]:

print "min=",min(train.humidity),"max=",max(train.humidity)


# In[69]:

mean(train.humidity)


# In[71]:

len(train['humidity'].unique())


##### So out of 101 integer values possible for humidity , it can take 89 integer as its value 

##### So now we have a good understanding of all the variables and their distributions we can start our analysis 

### Dummy Variable Creation 

##### The characteristic variables which have a less number of categories could be tackled by the creation of dummy variables 

# In[73]:

train['season'].value_counts()


# In[75]:

train['season1']=0
train['season2']=0
train['season3']=0
train['season1'][train['season']==1]=1
train['season2'][train['season']==2]=1
train['season3'][train['season']==3]=1


##### Remember the number of dummy variables created for 1 variable are equal to the total number of distinct categories of that variable -1

##### So 3 dummy variables are created in the above case 

# In[76]:

train['weather'].value_counts()


##### 2 dummy variables in the above case as there are negligible amount of observations in case of weather =4 (only 1 obs.)

# In[77]:

train['weather1']=0
train['weather2']=0
train['weather1'][train['weather']==1]=1
train['weather2'][train['weather']==2]=1


##### For binary variables there is no need to make a dummy variable a they themselves act like 1 

# In[79]:

train.index[:4]


# In[84]:

train.index[len(train)-1]


# In[85]:

temp2=pd.date_range(train.index[0],train.index[len(train)-1],freq='H')


# In[86]:

temp2[:2]


# In[87]:

len(train)


# In[88]:

len(temp2)


# In[89]:

len(test)


# In[90]:

train_start=train.index[0]


# In[94]:

train_start.date()


# In[96]:

train_start.time()


##### So clearly its 12 midnight 1st january 2011 from where the train data starts from 

# In[98]:

train_end=train.index[len(train)-1]


# In[99]:

train_end.date()


# In[100]:

train_end.time()


##### so clearly we have the train data till 11 pm 19th december 2012 

# In[101]:

test_start=test.index[0]


# In[102]:

test_start.date()


# In[103]:

test_start.time()


##### The 20th to the end of the month dats is missing form each of the month that we have to predict 

##### Lets test this using our time series analysis skills 

##### Calculate the length of the train data given that you only know that the first 19 days till 11pm of the 19th day is in time series 

# In[105]:

start=train.index[0]
start


# In[106]:

end=pd.Timestamp('2011-01-19 23:00:00')


# In[107]:

train_month=pd.date_range(start,end,freq='H')
len(train_month)


# In[110]:

len(train_month)*12*2


# In[109]:

len(train)


##### So we clealy see that some of the timestamps are missing form the train data precisely , 10944-10886=58 timestamps

##### Now lets check the same thing for the test data 

# In[111]:

test.index[len(test)-1]


# In[112]:

total=pd.date_range(start,test.index[len(test)-1],freq='H')


# In[113]:

total


# In[114]:

len(total)-len(train_month)*24


##### So these are the total number of timestamps data that should be there in the test data 

# In[116]:

len(test)


##### So again some timestamps missing in the data . Number of timestamps missing  = 6600-6493=107

##### lets take the first month of the train data :

# In[196]:

train.index[0]


# In[197]:

m1=train[:'2011-01-31 23:00:00']


# In[198]:

len(m1)


##### Also take the 1st month of the test data 

# In[199]:

t1=test[:'2011-02-01 00:00:00']


# In[200]:

t1.index[len(t1)-1]


# In[201]:

m1.index[len(m1)-1]


# In[202]:

(m1['registered']).plot(style='b*')
(m1['casual']).plot(style='r*')
(m1['count']).plot(style='y*')


##### This has got a lot of points . Lets make a daily data and then represent it on the graph to search for any pattern 

# In[203]:

m1D=m1.resample('D')


# In[204]:

m1D['casual'].plot(style='r-')
m1D['registered'].plot(style='g-')
m1D['count'].plot(style='b-')


##### Lets make a secondary y axis for the casual count 

# In[205]:

m1D['casual'].plot(style='r-',secondary_y=True)
m1D['registered'].plot(style='g-')
m1D['count'].plot(style='b-')


##### Clearly there is big similarity between the registered and the total count but there is a kink in the casual count during the first day of the month and during the 16th of the month 

##### Lets see if the same pattern is seen for other months too 

# In[206]:

len(m1)


# In[207]:

train.index[431]


# In[208]:

m2=train['2011-02-01 00:00:00':'2011-02-19 23:00:00']


# In[209]:

m2D=m2.resample('D')


# In[210]:

m2D['casual'].plot(style='r-',secondary_y=True)
m2D['registered'].plot(style='g-')
m2D['count'].plot(style='b-')


##### Here again there is a constant relationship between the registered and the count data except that there are some kinks in the casual count dude to which the count increases more than the registered amount 

##### These kinks in the casual count maybe due to the holiday or a non working day 

# In[211]:

train.head(1)


# In[212]:

from scipy.stats.stats import pearsonr


##### Lets try to find the pearson correlation coefficient between the casual count and the other variables 

# In[213]:

pearsonr(train['casual'],train['holiday'])


##### The p value above is very low which is a good thing for our result .Clearly the correlation coefficient is too low which suggest there is not much sorrelation between the 2 variables 

##### Lets try to find this with another method 

# In[214]:

H1 = train[train['holiday']==1]


# In[215]:

H0 = train[train['holiday']==0]


# In[216]:

H1['casual'].mean()


# In[217]:

H0['casual'].mean()


##### Clearly there is a high average on the holidays 

##### Now for working day : 

# In[218]:

W1 = train[train['workingday']==1]


# In[219]:

W0 = train[train['workingday']==0]


# In[220]:

W1['casual'].mean()


# In[221]:

W0['casual'].mean()


##### Here the differene is even bigger 

##### Clearly more casual count on a non working day 

# In[222]:

pearsonr(train['holiday'],train['workingday'])


##### There is a negative correlation here which is obvious as there will be no working day on a holiday 

##### Lets see if there is a correlation between the registered count and the variables : 

# In[223]:

H1['registered'].mean()


# In[224]:

H0['registered'].mean()


##### This is in contradiction to the general expected result 

# In[225]:

W1['registered'].mean()


# In[226]:

W0['registered'].mean()


##### Again this is in contradiction . This can happen only if the people use that bike to commute to their workplace 

##### Lets perform segmentation - Make 4 segments - On the basis of holiday variable and the workingday variable :

# In[250]:

H1= train[train['holiday']==1]
H1.shape


# In[251]:

H0=train[train['holiday']==0]
H0.shape


# In[252]:

W1=train[train['workingday']==1]
W1.shape


# In[253]:

W0=train[train['workingday']==0]
W0.shape


# In[254]:

H1W1=H1[H1['workingday']==1]
H1W1.shape


# In[255]:

H1W0 = H1[H1['workingday']==0]
H1W0.shape


# In[256]:

H0W1 = H0[H0['workingday']==1]
H0W1.shape


# In[257]:

H0W0 = H0[H0['workingday']==0]
H0W0.shape


##### So now that we have made 4 segments we can make models for each segment and apply it accordingly to the test DF 

# In[263]:

H1W0M1=H1W0['2011-01']
H1W0M1.shape


# In[266]:

ax1=H1W0M1['casual'].plot(style='r-',secondary_y=True)
ax2=H1W0M1['registered'].plot(style='g-')
ax2=H1W0M1['count'].plot(style='b-')
ax1.set_ylabel('Casual_count')
ax2.set_ylabel('Registered / Total count')


##### We clearly now notice the pattern is almost similar for all the 3 kinds of counts 

# In[267]:

H1W0M2=H1W0['2011-02']
H1W0M2.shape


# In[274]:

H1W0M3=H1W0['2011-03']
H1W0M3.shape


# In[275]:

H1W0M4=H1W0['2011-04']
H1W0M4.shape


# In[276]:

ax1=H1W0M4['casual'].plot(style='r-',secondary_y=True)
ax2=H1W0M4['registered'].plot(style='g-')
ax2=H1W0M4['count'].plot(style='b-')
ax1.set_ylabel('Casual_count')
ax2.set_ylabel('Registered / Total count')


# In[277]:

H1W0M5=H1W0['2011-05']
H1W0M5.shape


# In[278]:

H1W0M6=H1W0['2011-06']
H1W0M6.shape


# In[279]:

H1W0M7=H1W0['2011-07']
H1W0M7.shape


# In[281]:

ax1=H1W0M7['casual'].plot(style='r-',secondary_y=True)
ax2=H1W0M7['registered'].plot(style='g-')
ax2=H1W0M7['count'].plot(style='b-')
ax1.set_ylabel('Casual_count')
ax2.set_ylabel('Registered / Total count')


##### Thus we see that the same kind of pattern is same for all the 3 kinds of counts .

##### Thus our segmentation is correct and now we can model for all the 4 segments separately 

# In[284]:

train.columns


### Random Forests 

# In[353]:

from sklearn import ensemble


# In[354]:

clf = ensemble.RandomForestClassifier(n_estimators = 1000, n_jobs = -1)


# In[355]:

clf.fit(H1W0[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']],H1W0['count'])


# In[356]:

test=pd.read_csv('test.csv',parse_dates=['datetime'],index_col='datetime')


# In[357]:

test.columns


# In[358]:

test['season'].value_counts()


# In[359]:

test['weather'].value_counts()


# In[360]:

test['season1']=0
test['season2']=0
test['season3']=0
test['season1'][test['season']==1]=1
test['season2'][test['season']==2]=1
test['season3'][test['season']==3]=1


# In[361]:

test['weather1']=0
test['weather2']=0
test['weather1'][test['weather']==1]=1
test['weather2'][test['weather']==2]=2


# In[362]:

t_H1=test[test['holiday']==1]
t_H1.shape


# In[363]:

t_H0=test[test['holiday']==0]
t_H0.shape


# In[364]:

t_W1=test[test['workingday']==1]
t_W1.shape


# In[365]:

t_W0=test[test['workingday']==0]
t_W0.shape


# In[366]:

t_H1W0=t_H1[t_H1['workingday']==0]
t_H1W0.shape


# In[367]:

t_H1W1=t_H1[t_H1['workingday']==1]
t_H1W1.shape


# In[368]:

t_H0W0=t_H0[t_H0['workingday']==0]
t_H0W0.shape


# In[369]:

t_H0W1=t_H0[t_H0['workingday']==1]
t_H0W1.shape


##### So now we have divided the test DF also into 4 segments 

# In[370]:

t_H1W0.columns


# In[371]:

p_H1W0=clf.predict(t_H1W0[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']])


##### Similarly we can apply this model to other segments as well keeping the same number of estimators 

# In[373]:

clf.fit(H0W0[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']],H0W0['count'])


# In[374]:

p_H0W0=clf.predict(t_H0W0[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']])


# In[377]:

clf.fit(H0W1[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']],H0W1['count'])


# In[379]:

p_H0W1=clf.predict(t_H0W1[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']])


##### Now we need to just combine the 3 predicted dfs into the test data to make the submission 

# In[380]:

get_ipython().system(u'ls')


# In[396]:

ss=pd.read_csv('ss.csv')


# In[397]:

ss.head(2)


# In[398]:

p_H0W1


# In[399]:

p_H0W0


# In[400]:

p_H1W0


# In[401]:

ss=pd.read_csv('ss.csv',parse_dates=['datetime'],index_col='datetime')


# In[409]:

ss['count'][t_H0W0.index]=p_H0W0


# In[410]:

ss['count'][t_H0W1.index]=p_H0W1


# In[411]:

ss['count'][t_H1W0.index]=p_H1W0


# In[421]:

ss['datetime']=0


# In[422]:

ss['datetime']=ss.index


# In[425]:

ss.to_csv('segment4.csv',index=False)


##### We see we get a very bad score with this model . So we will try the regression technique instead of the random forests 

### Multilinear regression 

# In[456]:

from sklearn import  linear_model


# In[457]:

regr = linear_model.LinearRegression()


# In[458]:

regr.fit(H0W1[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']],H0W1['count'])


# In[459]:

regr.coef_


# In[460]:

p_H0W1=regr.predict(t_H0W1[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']])
p_H0W1


# In[461]:

p_H0W1.astype(int)


# In[462]:

p_H0W1=p_H0W1.astype(int)


# In[463]:

len(p_H0W1)


# In[464]:

len(t_H0W1)


# In[465]:

regr.fit(H0W0[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']],H0W0['count'])


# In[466]:

p_H0W0=regr.predict(t_H0W0[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']])
p_H0W0=p_H0W0.astype(int)


# In[467]:

len(p_H0W0)


# In[468]:

regr.fit(H1W0[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']],H1W0['count'])


# In[469]:

p_H1W0=regr.predict(t_H1W0[['humidity','windspeed','season1','season2','season3','weather1','weather2','temp','atemp']])
p_H1W0=p_H1W0.astype(int)


# In[470]:

len(p_H1W0)


# In[471]:

p_H1W0


# In[472]:

p_H0W0[p_H0W0<1]=1
p_H0W1[p_H0W1<1]=1
p_H1W0[p_H1W0<1]=1


# In[473]:

p_H1W0


# In[474]:

ss=pd.read_csv('ss.csv',parse_dates=['datetime'],index_col='datetime')


# In[475]:

ss['count'][t_H0W0.index]=p_H0W0
ss['count'][t_H0W1.index]=p_H0W1
ss['count'][t_H1W0.index]=p_H1W0
ss['datetime']=0
ss['datetime']=ss.index
ss.to_csv('Regression.csv',index=False)


# In[ ]:



