
# coding: utf-8

## Time Series Analysis using Pandas 

### Kaggle Case Study : Bike Sharing Demand 

# In[454]:

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import math
import os


##### Os module is used for directory operations 

# In[199]:

os.getcwd()


# In[201]:

get_ipython().system(u'ls')


# In[203]:

pd.__version__


##### We are using pandas version 0.14.1

# In[204]:

os.chdir('archive')


# In[205]:

get_ipython().system(u'ls')


##### The following statement is used so that all the graphs and plots are displayed within the notebook

# In[206]:

get_ipython().magic(u'pylab inline')


##### For example: 

# In[207]:

plt.plot(np.arange(100))


##### Thus the plot is displayed within the notebook 

#### New datatype in pandas for datetime .Its very efficient and fast . Really good for millions of observations data (very large scale time series processing )

# In[208]:

plt.plot(np.random.randn(1000).cumsum())


#### This code generates a series of 1000 random numbers and calculate its cumulative sum column , so we get this decreasing graph 

## Some REVISION

# In[209]:

np.arange(5)[2:]#slicing


# In[210]:

index=['a','b','c','d','e']


# In[211]:

s=Series(np.arange(5),index=index)


# In[212]:

s


##### So this is a vertical 1 dimentional series with labels/Index

##### Series contains labels that identify the data

##### Series object behaves much like a numpy array 

# In[213]:

s*2


# In[214]:

s+s


# In[215]:

s/s


#### slicing :

# In[216]:

s[2:3]


# In[217]:

s['b']#value from labels


# In[218]:

s['b':]#slicing with labels


# In[219]:

s[[2,4]]


# In[220]:

s[['c','e']]


##### These were the ways of slicing we could do 

# In[221]:

dates=pd.date_range('2012-07-16','2012-07-21')
dates


##### This is a DateTimeIndex variable array created 

# In[222]:

atemps=Series([101.5,98,95,99,100,92],index=dates)


# In[223]:

atemps


# In[224]:

atemps.index


# In[225]:

idx=atemps.index[2]
atemps[idx]


##### So we can use this way to find the temp at that day

# In[226]:

sdtemps=Series([73,78,77,78,78,77],index=dates)
temps=DataFrame({'Austin':atemps,'San Diago':sdtemps})
temps


##### This is how we make a dataframe using a dictionary inside the dataframe function 

# In[227]:

temps['Austin']#to get the austin column


# In[228]:

temps['San Diago'][:3]


##### If now we want the difference of 2 columns:

# In[229]:

temps['diff']=temps['Austin']-temps['San Diago']


# In[230]:

temps


##### So there are a lot of ways we can play with this data using these dataframes

##### We can also call a column name using . operator 

# In[231]:

temps.Austin


##### Now remember the series , we can do the same kind of indexing and slicing in dataframe

##### suppose we want the variable values for a particular observation that is the temperature values at a particular date at both the places 

# In[232]:

temps.ix[2]


# In[233]:

del temps['diff']


# In[234]:

temps.ix[[1,0,2],['San Diago','Austin']]


##### now making a timestamp and using it in slicing and indexing :

# In[235]:

idx=temps.index[2]


# In[236]:

type(idx)


##### This is a Timestamp

# In[237]:

temps.ix[idx]


# In[238]:

temps


##### Now suppose we want to do operations on variables like calculating the mean temperature at both the places 

# In[239]:

temps.mean()


#### The describe function 

# In[240]:

temps['Austin'].describe()


# In[241]:

temps.describe()


##### We can have intelligent computations like deviation from mean value at each date 

# In[242]:

temps-temps.mean()


##### Now suppose we want the subset where temp at austin > 100 

# In[243]:

temps[temps.Austin>100]


##### So there was only 1 date when temperature at Austin is >100

# In[244]:

get_ipython().system(u'ls')


##### Cat function is used to see the kind of file what is there in the file 

# In[246]:

get_ipython().system(u'ls')


# In[247]:

get_ipython().system(u'cat test1.csv')


##### So this basically has some data with index as a date time data type 

##### lets read this file in a dataframe 

# In[248]:

test1=pd.read_csv('test1.csv')


# In[249]:

test1.head(1)


# In[250]:

type(test1.date[0])


##### So this just takes it as a string , we need to change it to datetime variable so that we can perform operations on it 

# In[251]:

test1=pd.read_csv('test1.csv',parse_dates=['date'])


##### This converts the values in the date column to python datetime objects

# In[252]:

test1.date


##### So clearly we get back a python datetime object 

# In[253]:

type(test1.date[0])


##### So cleary this time we get time stamps in the index 

##### Now we notice one thing that the index is no the date column , if we want the date to be index of our dataframe , we need to specify it 

# In[254]:

test1=pd.read_csv('test1.csv',parse_dates=['date'],index_col='date')


# In[255]:

test1.head(3)


##### So clearly we get date as the index now 

# In[256]:

test1.index.name


# In[257]:

test1.index


##### SO we get back here as a datetimeindex column 

##### Now suppose we want the date column as a variable as well as the index of the dataframe 

# In[258]:

test1=pd.read_csv('test1.csv',parse_dates=['date'])


# In[259]:

test1.head(1)


# In[260]:

test1.set_index('date',drop=False).head(2)


##### So here we can still access the date and the index is also a date 

# In[261]:

test2=pd.read_csv('test2.csv')


# In[262]:

test2.head(2)


##### Here clearly the date and the time are splitted in 2 columns 

##### Suppose we want to combine date and time column and then parse it 

# In[263]:

test2=pd.read_csv('test2.csv',parse_dates=[['date','time']])


# In[264]:

test2.head(2)


# In[265]:

test2.date_time[0]


##### so it has clearly glued the 2 columns together and parsed them as a timestamp

# In[266]:

test2=pd.read_csv('test2.csv',parse_dates=[['date','time']],
                  index_col=['date_time'])


# In[267]:

test2.head(2)


# In[268]:

test2.index


##### We use a question mark to know about a particular function 

# In[269]:

get_ipython().magic(u'pinfo pd.read_csv')


#### So now we will learn mode about the Datetime data variable 

###### datetime64[ns]:nanosecond , this was added to numpy in about 2010 . Each time stamp represens ~600 year timestamp , 1700-2300

##### Datetimeindex - supports for duplicate timestamp . Timestamp is a subclass of datetime that supports nanoseconds 

# In[270]:

pd.Timestamp


# In[271]:

pd.Timestamp.mro()


##### Making a timestamp : 

# In[273]:

stamp=pd.Timestamp('7/17/2012 10:06:45')


# In[274]:

stamp.value #this gives the nanosecond value 


# In[275]:

print stamp.second,stamp.year,stamp.month,stamp.day,stamp.hour


# In[276]:

dates=['7/16/2012','7/18/2012','7/20/2012']
dates


# In[277]:

s1=Series(range(3),index=dates)
s1


# In[278]:

s1.index


##### This is how we converted the dates to datetimeindex

# In[279]:

pd.DatetimeIndex(dates)


##### This is how to convert directly into datetimeindex

#### Dayfirst option in datetime

# In[280]:

dates=['7/09/2012','7/10/2012','7/11/2012']
dates


# In[281]:

list(pd.to_datetime(dates,dayfirst=True))


##### So here we get the European format instead of the US format of date 

# In[282]:

s1.index[0]


# In[283]:

s1


#### Lets talk about time series indexing 

## Indexing

# In[284]:

rng=pd.date_range('2012-07-17',periods=1000)


##### Here we create fix frequency of timestamps as a datetime object 

# In[285]:

rng


##### Creating a time series now :

# In[286]:

ts=Series(np.arange(1000),index=rng)


# In[287]:

ts[:3]


# In[288]:

ts[997:]


##### so we get the above time series now . Now suppose we want to select just a single value from the above time series 

##### By indexing we mean selecting elements from the time series 

# In[289]:

ts[ts.index[133]]


##### We can also pass the string 

# In[290]:

ts['2012-11-27']


##### We want to select data upto and including 2012-11-27

# In[291]:

ts[:'2012-11-27']


##### The end point is generally not included in case of numpy array and series slicing but it is included incase of a time series 

##### for example: 

# In[292]:

a=np.arange(100)


# In[293]:

a[:10]


##### only till 9 are displayed not the 10th one 

# In[294]:

rng=pd.date_range('2012-07-17',periods=1000)


# In[295]:

rng


# In[296]:

rng[0]


# In[297]:

rng[1]


##### observe how the date is increasing step by step 

##### Notice that the offset is D so the data is increasing day by day 

# In[298]:

ts2=ts.take(np.random.permutation(len(ts)))


# In[299]:

ts2


# In[300]:

ts2.sort_index()


##### sort_index returns a time series sorted by date 

# In[301]:

ts['2012-7-26']


##### we can also select subsets of the data like only july 2012 

# In[302]:

ts['2012-7']


# In[303]:

ts['2012-7':'2012-8']


##### This only gives data of july and aug 

##### similarly we could do it for year also 

# In[304]:

df1=DataFrame(np.arange(4000).reshape((1000,4)),index=rng)


# In[305]:

df1.head(1)


##### we can also name the columns

# In[306]:

df1=DataFrame(np.arange(4000).reshape((1000,4)),index=rng,
              columns=['delhi','mumbai','calcutta','tamil'])


# In[307]:

df1.head(1)


# In[308]:

df1.ix['2012-7-29']


##### So using ix we get the data for all the cities for a particular date 

# In[309]:

df1.ix['2012-7-29'].index


# In[310]:

df1.ix['2012-7-20'].name


##### This gives us the timestamp

# In[311]:

from datetime import timedelta,datetime


# In[312]:

start=datetime(2012,12,31)
df1.ix[start:start+timedelta(days=11)]


# In[313]:

len(df1.ix[start:start+timedelta(days=11)])


##### so we get here a data of 12 days not 11 as the end point is also included in the case of a time series 

##### The above thing could also be done using pandas offsets object 

# In[314]:

df1.ix[start:start+pd.offsets.Day(11)]


##### The kind of offsets in time series 

##### D: for a calender day .B is for a business day . M is calender end of the month . BM is business end of month . 

# In[315]:

list(pd.date_range('2000-01-01',periods=5))


##### This is a timeseries generated with a frequency of a calender day 

##### With business day it is the weekends which are not considered 

##### H is for an hour , s is for a second 

# In[316]:

list(pd.date_range('2000-01-01',periods=5,freq='BM'))


##### This is used in case of a financial calender 

##### MS is for start of the month 

# In[317]:

list(pd.date_range('2000-01-01',periods=5,freq='H'))


# In[318]:

list(pd.date_range('2000-01-01',periods=5,freq='s'))


# In[319]:

list(pd.date_range('2000-01-01',periods=5,freq='m'))


##### m is for the month end not minutes 

##### Annual dates falling on last day of june 

# In[320]:

list(pd.date_range('2000-01-01',periods=5,freq='BA-FEB'))


##### for quaterly dates 

# In[321]:

list(pd.date_range('2000-01-01',periods=5,freq='Q-DEC'))


##### for 3rd friday of each month 

# In[322]:

list(pd.date_range('2000-01-01',periods=5,freq='WOM-2FRI'))


# In[323]:

atemps=temps.Austin


# In[325]:

sdtemps=temps['San Diago'].drop(temps.index[3])


# In[327]:

atemps


# In[328]:

sdtemps


##### 1 less record here 

# In[329]:

temps-temps[:-1]


##### This gives NAN in the last row as there is no last row in the temp[:-1]

# In[330]:

diff=atemps-sdtemps
diff


# In[331]:

diff.fillna(method='ffill')


##### This fills the na value with the previous value . Similarly bfill

# In[332]:

diff.fillna(999)


## Resampling 

#### Resampling is done both in the cases of very high frquency time series as well as very low frequency time series . The high frequency time series is converted to low frequency time series and vice versa . The terms used are upsampling and the downsampling .

# In[336]:

datetime.now()


##### This gibes the timestamp of the present time and date 

# In[339]:

rng=pd.date_range(datetime.now(),periods=1000000,freq='t')


# In[340]:

s=Series(np.random.randn(1000000),index=rng)


# In[341]:

s[s.index[:15]]


# In[344]:

s.ix[s.index[:5]]


##### API method called reindex

# In[345]:

s.reindex(s.index[:15])


##### Select out the data every 30 minutes instead of every minute 

# In[349]:

rng_resampled=pd.date_range(rng[0],rng[-1],freq='30t')


##### The above definition produces the same time series as the original one but with a frquency of 30 minutes instead of a minute 

# In[350]:

len(rng_resampled)


# In[351]:

len(rng)


# In[353]:

float(len(rng)/len(rng_resampled))


# In[365]:

s[rng_resampled][:3]


# In[356]:

s_new=s[rng_resampled]


# In[369]:

s_new[:3]


# In[383]:

s[:3]


#### Now we will use the Resample () function to do the same function but with more control and flexibity

# In[386]:

s_resampled=s.resample('30t')
s_resampled[:3]


# In[387]:

s[:3]


##### Now we have some aggregate functions as to how we want to resample the original sample . For this we use an attribute 

# In[389]:

s_count=s.resample('30t',how='count')
s_count[:5]


##### So clearly it puts 19 samples in the first bin

##### by default the mean is calculated if no function is given , we can see that in the following way

# In[391]:

len(s[:19])


# In[392]:

mean(s[:19])


# In[394]:

s_resampled[0]


##### Therefore clearly by default the mean value is calculated for a bin . Now other aggregrate functions that we can perform are : 

# In[396]:

s_sum=s.resample('h',how='sum')
s_sum[:3]


# In[399]:

sum(s[:49]) #checking 


##### Now another option which we can give is that which way should it be closed 

# In[400]:

s_count[:5]


# In[407]:

s_count_left=s.resample('30t',how='count',closed='left')
s_count_left[:5]


##### What this closed option does is controls the end points that is at which place should the time 10:30 be . If we give left then time 10:30 will be the first time in the second bin and if we give right then 10:30 will be the last time of the first bin  

# In[403]:

s_count_right=s.resample('30t',how='count',closed='right')
s_count_right[:5]


##### Now another option we give is the label . What should be the label of the first bin , Should it be the left most figure or the rightmost figure 

# In[409]:

s_count=s.resample('30t',how='count')
s_count[:3]


##### Clearly by default the label is the leftmost timestamp

# In[410]:

s_count=s.resample('30t',how='count',label='right')
s_count[:3]


##### now the labels are shifted 30 minutes ahead than the previous labels

# In[411]:

s_daily=s.resample('D',how='count')
s_daily[:5]


### Timedelta

# In[412]:

from datetime import timedelta


# In[413]:

s_resample=s.resample('30t',how='count',
                      loffset=timedelta(seconds=-1),
                      closed='left',label='right')


# In[414]:

s_resample[:3]


##### So here see the labels , they have an offset of 1 second 

##### Now resample to monthly by mean 

# In[415]:

s_month=s.resample('M',how='max')


# In[417]:

s_month[:3]


##### We can even put our own functions in the how option for example the np andmath actions 

# In[418]:

import timeit


# In[423]:

get_ipython().magic(u"timeit s.resample('M',how='max')")


# In[425]:

get_ipython().magic(u"timeit s.resample('M',how=np.max)")


##### We can also have more than 1 methods 

# In[426]:

resampled_s=s.resample('5M',how=['count','mean','max','min','sum'])


# In[427]:

resampled_s


##### Now we will see more of the functions that are generally used 

##### For a financial analyst , open high low close are 4 values that are very important and used in their analysis 

# In[428]:

sf=s.resample('D',how='ohlc')


# In[429]:

sf[:4]


# In[433]:

resampled=s.resample('5t',how='ohlc')
resampled=resampled[:3]
resampled


#### Down sampling

# In[435]:

ds=resampled.resample('t')
ds


##### Clearly there is no data available by default NAN is included

##### lets fill these values 

# In[437]:

ds.fillna(method='ffill')


##### This is a fill forward method where the previous value is used to fill the missing forward values 

##### We can also put a limit to this filling .

# In[438]:

ds.fillna(method='ffill',limit=2)


##### so ffill only takes place till 2 fills rest are still NAN

##### Like ffill , there is bfill ( again it is obvious what it means )

#### Interpolation 

# In[440]:

ds


# In[441]:

ds.interpolate()


##### This function used for series is used to interpolate the values in the column . Linearly the values are filled 

##### Now suppose for different columns we want to fill it with different methods and different limits . For this we need to make a dictionary first 

# In[447]:

limits={'open':4,'high':3,'low':2,'close':1}
limits


# In[446]:

ds[:5]


# In[449]:

ds.apply(lambda x: x.fillna(limit=limits[x.name],method='ffill'))


##### Now suppose wa want to fill the values using a method while doing the undersampling itself 

# In[450]:

resampled


# In[451]:

resampled.resample('2t',fill_method='bfill')


##### Thus the function used is the fill_method 

## Periods logic 

##### Periods are time spans vs time points (timestamps) .They help in easy calender arithmetic 

##### pandas has got a period type 

# In[456]:

from pandas import Period as Pe


# In[457]:

p=Pe('2011',freq='A-jun')


# In[458]:

p


##### Arithmetic 

# In[460]:

p+2


# In[462]:

p.year


##### Conversions - suppose to monthly 

# In[463]:

p.asfreq('M')


##### we get back june 2011 as it was A-june , annual june. If we want the start of the annual session then use : 

# In[464]:

p.asfreq('M',how='start')


##### We get july 2010 here .

##### Suppose we wanted the second month in the interval and third to last business day in that month 

# In[465]:

(p.asfreq('M',how='start')+1).asfreq('B','end')-2


##### B is for business day 

# In[471]:

s[:5]


# In[478]:

s_month=s.resample('M')
s_month[0]


# In[479]:

type(s_month[0])


# In[480]:

type(s_month)


# In[481]:

s.index[0]


##### The index of the series s is a timestamp . Suppose we need a period span instead of a time stamp.

# In[483]:

s_month=s.resample('M',kind='Period')


# In[486]:

s_month[:4]


# In[488]:

type(s_month.index)


##### So we get a period index now instead of a time stamp

# In[489]:

s_month=s_month[:4]


# In[491]:

list(s_month.index)


##### So we list the period objects now 

##### We can do the same frequency logics now 

# In[492]:

s_month.asfreq('B',how='end')


##### Again to make a period range we use the same function as date_range

# In[493]:

pd.period_range('2012-07',periods=3,freq='M')


##### Starts in july and has 3 periods 

# In[495]:

get_ipython().system(u'ls')


# In[496]:

md=pd.read_csv('macrodata.csv')


# In[497]:

md[:4]


# In[498]:

md.shape


##### We have been year and a quarter 

# In[500]:

md.year[:4]


# In[504]:

ind=pd.PeriodIndex(year=md.year,quarter=md.quarter,freq='Q-DEC')
ind


##### Clearly it is a quarterly data that ends in december 

# In[505]:

md.index=ind


# In[506]:

md[:4]


# In[507]:

md.index


##### We can drop out the first 2 columns now 

# In[508]:

del md['year']
del md['quarter']


# In[510]:

md.head(4)


##### Now lets make time series plots 

# In[511]:

plt.plot(md.unemp)


##### Lets make the above data only for 2008 

# In[512]:

md.unemp['2008':].plot()


# In[513]:

md.unemp[:'1960'].plot()


##### Multiple datas plotting 

# In[514]:

md.ix['2008':,['unemp','tbilrate']].plot()


##### Suppose we want to plot these on separate axis , we use the following way 

# In[519]:

ax1=md.ix['2008':,'infl'].plot(style='k')
ax2=md.ix['2008':,'cpi'].plot(secondary_y=True,style='g')
ax1.set_ylabel('inflation')
ax2.set_ylabel('cpi')


##### Inflation scale is the left scale and the black line and the right scale represents the CPI and the green line represents it 

## Time Zone Handling 

#### There is no time zone till now that was asoociated with out working 

# In[520]:

pd.Timestamp('now')


##### This is the present timestamp 

# In[525]:

stamp_naive=pd.Timestamp('2012-07-17 11:00')#no time zone here 
stamp_naive


# In[526]:

stamp=pd.Timestamp('2012-07-17 11:00',tz='US/Central')
stamp


# In[527]:

stamp.tz


##### So now the time stamp has time zone info as well

##### Converting into a time zone a naive time zone

# In[528]:

stamp_naive


# In[529]:

stamp2=stamp_naive.tz_localize('US/Central')


# In[530]:

stamp2


##### Now I am In India lets do it form our point of view 

# In[535]:

stamp=pd.Timestamp('now')
stamp


##### We in general get the time stamp with respect to the GMT ...not the computer time 

##### Now as India is GMT+5:30 , we will just add that much time to create a timestamp of the present time in india and then we will initialize the timezone of india

# In[546]:

from datetime import datetime
stamp=pd.Timestamp('now')
stamp_ind=datetime(stamp.year,stamp.month,stamp.day,stamp.hour+5
                    ,stamp.minute+30,stamp.second)
stamp_ind


# In[548]:

stamp_ind=pd.Timestamp(stamp_ind)
stamp_ind


##### This is the correct present Indian time . We will now convert it to the the time in US by first initializing it as the indian time and then convertion 

# In[555]:

stamp_ind=stamp_ind.tz_localize('Asia/Kolkata')
stamp_ind


##### So now we can convert this to US time using the tz_convert function 

# In[558]:

stamp_us=stamp_ind.tz_convert('US/Central')
stamp_us


##### So there is 11:15 am in US at this time 

# In[559]:

stamp_ind.value


# In[560]:

stamp_us.value


##### So they have the same value . So 2 time zones of same time stamp could be compared by comparing their values 

# In[563]:

rng=pd.date_range('2014-10-01',periods=5,tz='Asia/Kolkata')
s_tz=Series(range(5),index=rng)
s_tz


#### UTC

##### UTC is from where all the other time zones are referenced as 

##### Suppose we want the time now in UTC so we use the following code for that:

# In[584]:

N=pd.Timestamp('now')


# In[585]:

N


# In[586]:

N=N.tz_localize('utc')
N


##### So the present UTC time is 4:39 pm
