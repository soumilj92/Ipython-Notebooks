
# coding: utf-8

# In[6]:

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
from datetime import datetime
import random
from sklearn import ensemble
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


# In[2]:

get_ipython().magic(u'pylab inline')
os.chdir('desktop/clicks')

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
ss=pd.read_csv('sampleSubmission.csv')


# In[3]:

import random
rnd=random.sample(range(len(train)),1000000)
rnd1=rnd[:500000]
rnd2=rnd[500000:]
t1=train.ix[rnd1]
t2=train.ix[rnd2]


# In[4]:

t1['h']=0
t1['d']=0
t2['h']=0
t2['d']=0
test['h']=0
test['d']=0
t1.index=range(500000)
t2.index=range(500000)

for i in range(len(t1)):
    t1['h'][i]=int(str(t1['hour'][i])[6:8])
    t1['d'][i]=int(str(t1['hour'][i])[4:6])
    t2['h'][i]=int(str(t2['hour'][i])[6:8])
    t2['d'][i]=int(str(t2['hour'][i])[4:6])
    
for i in range(len(test)):
    test['h'][i]=int(str(test['hour'][i])[6:8])
    test['d'][i]=int(str(test['hour'][i])[4:6])


# In[7]:

t1['weekday']=0
t2['weekday']=0
test['weekday']=0
for i in range(len(t1)):
    t1['weekday'][i]=pd.Timestamp(datetime(2014,10,t1['d'][i],t1['h'][i])).dayofweek
    t2['weekday'][i]=pd.Timestamp(datetime(2014,10,t2['d'][i],t2['h'][i])).dayofweek
for i in range(len(test)):
    test['weekday'][i]=pd.Timestamp(datetime(2014,10,test['d'][i],test['h'][i])).dayofweek


# In[309]:

def validate(f) : 
    model = LogisticRegression()
    model = model.fit(t1[f],t1['click'])
    print 'model_score in time validation >>>>>',model.score(t1[f],t1['click'])
    print 'model_score out of time validation >>>>>',model.score(t2[f],t2['click'])
    probs = model.predict_proba(test[f])
    predictY= DataFrame(probs)
    ss['click']=predictY[1]
    ss.to_csv('new_way.csv',index=False)


# In[88]:

get_ipython().magic(u'pinfo plt.text')


# In[103]:

def sorted_plot(f,set1,k):
    temp_avg=np.array([])
    for i in range(len(set1[f].unique())):
        temp_avg=np.append(temp_avg,np.mean(set1['click'][set1[f]==(set1[f].unique())[i]]))
    if(k==2):
        f1=figure()
        plt.plot(range(len(set1[f].unique())), temp_avg, 'bo',range(len(set1[f].unique())), temp_avg, 'k')
        plt.grid()
        plt.show()
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    df_temp=DataFrame({f:set1[f].unique(),'Avg_click':temp_avg})
    df_temp=df_temp.sort(columns='Avg_click')
    plt.plot(range(len(set1[f].unique())), df_temp['Avg_click'], 'bo',range(len(set1[f].unique())), df_temp['Avg_click'], 'k')
    for x,y in zip(range(len(set1[f].unique())), df_temp['Avg_click']):                                                # <--
        #ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='offset points')
        plt.text(x-0.004,y+0.007,df_temp[f][x],fontsize=12)
    plt.grid()
    plt.show()
    print 'total count =',len(df_temp)
    print 'maximum_value =',df_temp['Avg_click'].max(),'at',df_temp[f][df_temp['Avg_click']==df_temp['Avg_click'].max()]
    print 'minimum_value =',df_temp['Avg_click'].min(),'at',df_temp[f][df_temp['Avg_click']==df_temp['Avg_click'].min()]
    print 'number of zeroes =',len(df_temp[df_temp['Avg_click']==0])
    print 'number of ones =',len(df_temp[df_temp['Avg_click']==1])
    print 'Distribution:(including the starting value)'
    for i in np.arange(0,1.1,0.1):
        print i,'<->',(i+0.1),' = ',len(df_temp[(df_temp['Avg_click']>=i) & (df_temp['Avg_click']<(i+0.1))])


# In[10]:

def validation_check(feature_cols,list1):
    validate(feature_cols)
    print 'After Adding : '
    new=feature_cols+list1
    validate(new)


# In[11]:

t1.columns


# In[12]:

site=train['site_category'].value_counts()
site=site[site>30]
site=site.index


# In[13]:

avg_site=np.array([])
for i in site:
    avg_site=np.append(avg_site,np.mean(train['click'][train['site_category']==i]))


# In[14]:

site_df=DataFrame({'site':site,'avg_click':avg_site})
site_df=site_df.sort(columns='avg_click')
plt.plot(range(len(site_df)), site_df['avg_click'], 'bo',range(len(site_df)), site_df['avg_click'], 'k')


# In[17]:

site_df.tail(2)


# In[21]:

t1['special_site']=0
t2['special_site']=0
test['special_site']=0
t1['special_site'][t1['site_category']=='dedf689d']=1
t2['special_site'][t2['site_category']=='dedf689d']=1
test['special_site'][test['site_category']=='dedf689d']=1


# In[22]:

print sum(t1['special_site']),sum(t2['special_site']),sum(test['special_site'])


# In[23]:

feature_cols=['special_site']


# In[24]:

validate(feature_cols)


# In[25]:

sorted_plot('C1',train,1)


# In[32]:

c1=train['C1'].unique()
avg_c1=np.array([])
for i in c1:
    avg_c1=np.append(avg_c1,np.mean(train['click'][train['C1']==i]))


# In[33]:

df_c1=DataFrame({'c1':c1,'avg_click':avg_c1})


# In[34]:

plt.plot(df_c1['avg_click'])


# In[35]:

df_c1=df_c1.sort(columns='avg_click')
plt.plot(range(len(df_c1)), df_c1['avg_click'], 'bo',range(len(df_c1)), df_c1['avg_click'], 'k')        


# In[36]:

df_c1


# In[37]:

t1['C1'][0]


# In[40]:

l1=[1001,1007]
l2=[1010,1008]
l3=[1005,1012]
t1['c1_1']=0
t2['c1_1']=0
test['c1_1']=0
t1['c1_2']=0
t2['c1_2']=0
test['c1_2']=0
t1['c1_3']=0
t2['c1_3']=0
test['c1_3']=0
for k in l1:
    t1['c1_1'][t1['C1']==k]=1
    t2['c1_1'][t2['C1']==k]=1
    test['c1_1'][test['C1']==k]=1
for k in l2:
    t1['c1_2'][t1['C1']==k]=1
    t2['c1_2'][t2['C1']==k]=1
    test['c1_2'][test['C1']==k]=1
for k in l3: 
    t1['c1_3'][t1['C1']==k]=1
    t2['c1_3'][t2['C1']==k]=1
    test['c1_3'][test['C1']==k]=1


# In[41]:

l=['c1_1','c1_2','c1_3']


# In[42]:

validation_check(feature_cols,l)


# In[49]:

print sum(test['c1_1']),sum(test['c1_2']),sum(test['c1_3'])


# In[45]:

feature_cols


# In[50]:

#Now lets try to check the C1 variable with 6 dummy variables 


# In[52]:

c1=t1['C1'].unique()
for i in range(1,7,1):
    pair='c1',format(i)
    str1=''.join(pair)
    t1[str1]=0
    t2[str1]=0
    test[str1]=0
    t1[str1][t1['C1']==c1[i-1]]=1
    t2[str1][t2['C1']==c1[i-1]]=1
    test[str1][test['C1']==c1[i-1]]=1


# In[54]:

print sum(test['c11']),sum(test['c12']),sum(test['c13']),sum(test['c14']),sum(test['c15']),sum(test['c16'])


# In[55]:

test['C1'].value_counts()


# In[56]:

train['C1'].value_counts()


# In[57]:

#1008 wont make a difference in case of the test data 


# In[58]:

c1


# In[61]:

c1=train['C1'].value_counts()
c1=c1.index
c1


# In[62]:

for i in range(1,7,1):
    pair='c1',format(i)
    str1=''.join(pair)
    t1[str1]=0
    t2[str1]=0
    test[str1]=0
    t1[str1][t1['C1']==c1[i-1]]=1
    t2[str1][t2['C1']==c1[i-1]]=1
    test[str1][test['C1']==c1[i-1]]=1


# In[63]:

print sum(test['c11']),sum(test['c12']),sum(test['c13']),sum(test['c14']),sum(test['c15']),sum(test['c16'])


# In[64]:

l_c1=['c11','c12','c13','c14','c15','c16']


# In[65]:

validation_check(feature_cols,l_c1)


# In[66]:

feature_cols


# In[68]:

validate(l_c1)


### LEFT C1

# In[69]:

t1.columns


# In[71]:

train['banner_pos'].value_counts()


# In[72]:

test['banner_pos'].value_counts()


###### There is no banner position called 5 in the test data 

# In[115]:

sorted_plot('banner_pos',train,1)


# In[107]:

t1['banner_high']=0
t2['banner_high']=0
test['banner_high']=0
t1['banner_high'][t1['banner_pos']==3]=1
t2['banner_high'][t2['banner_pos']==3]=1
test['banner_high'][test['banner_pos']==3]=1


# In[108]:

l_banner_high=['banner_high']


# In[109]:

validate(l_banner_high)


# In[110]:

validation_check(feature_cols,l_banner_high)


# In[111]:

#Making banner bins


# In[112]:

test['banner_pos'].value_counts()


# In[114]:

train['banner_pos'].value_counts()


# In[116]:

l_banner_low=['banner_low']


# In[118]:

t1['banner_low']=0
t2['banner_low']=0
test['banner_low']=0
t1['banner_low'][(t1['banner_pos']==1) | (t1['banner_pos']==0)]=1
t2['banner_low'][(t2['banner_pos']==1) | (t2['banner_pos']==0)]=1
test['banner_low'][(test['banner_pos']==1) | (test['banner_pos']==0)]=1


# In[123]:

validation_check(feature_cols,(l_banner_low+l_banner_high))


# In[124]:

t1.columns


# In[125]:

sorted_plot('h',t1,1)


# In[126]:

#all are between 1 and 2 not much difference , This could be put as a linear variable by changing the values 


# In[127]:

h_ascend=[19,6,9,1,14,12,15,13,11,17,10,7,4,0,8,2,21,3,5,16,18,20,23,22]


# In[128]:

len(h_ascend)


# In[ ]:




# In[154]:

h_df=DataFrame({'rank':range(1,25,1),'h':h_ascend})
A_df=h_dict[h_dict['h']==14]
A_df.iloc[0]['rank']


# In[155]:

h_df=DataFrame({'rank':range(1,25,1),'h':h_ascend})
t1['h_rank']=0
t2['h_rank']=0
test['h_rank']=0
for i in range(len(t1)):
    A_df=h_df[h_df['h']==t1['h'][i]]
    t1['h_rank'][i]=A_df.iloc[0]['rank']


# In[156]:

h_df=DataFrame({'rank':range(1,25,1),'h':h_ascend})
for i in range(len(t2)):
    A_df=h_df[h_df['h']==t2['h'][i]]
    t2['h_rank'][i]=A_df.iloc[0]['rank']


# In[157]:

for i in range(len(test)):
    A_df=h_df[h_df['h']==test['h'][i]]
    test['h_rank'][i]=A_df.iloc[0]['rank']


# In[162]:

validate(['h_rank']+feature_cols)


# In[160]:

print feature_cols


# In[163]:

validation_check(feature_cols,['h_rank'])


# In[164]:

t2['click'].value_counts()


# In[176]:

def validate2(f):
    model = LogisticRegression()
    model = model.fit(t1[f],t1['click'])
    probs = model.predict_proba(t2[f])
    predictY= DataFrame(probs)
    p=predictY[1]
    s=0
    for i in range(len(t2)):
        s=s+abs((t2['click'][i])-p[i])
    print s


# In[177]:

validate2(feature_cols)


# In[182]:

validate2(['banner_low'])


# In[183]:

validate2(['banner_high'])


# In[184]:

validate2(l_c1)


# In[185]:

c1


###### l_c1 seems to be a better estimator than feature_cols

# In[186]:

print feature_cols


# In[187]:

validate2(l_c1+feature_cols)


# In[190]:

feature_cols=['special_site']+l_c1


# In[191]:

print feature_cols


# In[192]:

t1.columns


# In[201]:

def validation_check2(f,l):
    validate2(f)
    print '-----'
    validate2(l)
    print '-----'
    validate2(l+f)


# In[194]:

print feature_cols


# In[195]:

sorted_plot('device_conn_type',train,1)


# In[196]:

conn=train['device_conn_type'].unique()
conn


# In[198]:

test['device_conn_type'].value_counts()


# In[199]:

for i in range(1,4,1):
    pair='conn_type',format(i)
    str1=''.join(pair)
    t1[str1]=0
    t2[str1]=0
    test[str1]=0
    t1[str1][t1['device_conn_type']==conn[i-1]]=1
    t2[str1][t2['device_conn_type']==conn[i-1]]=1
    test[str1][test['device_conn_type']==conn[i-1]]=1


# In[202]:

validation_check2(feature_cols,['conn_type1','conn_type2','conn_type3'])


# In[203]:

t1.columns


# In[206]:

test['C18'].value_counts()


# In[207]:

sorted_plot('C18',train,1)


# In[208]:

c18=train['C18'].unique()
for i in range(1,4,1):
    pair='c18',format(i)
    str1='_'.join(pair)
    t1[str1]=0
    t2[str1]=0
    test[str1]=0
    t1[str1][t1['C18']==c18[i-1]]=1
    t2[str1][t2['C18']==c18[i-1]]=1
    test[str1][test['C18']==c18[i-1]]=1


# In[209]:

feature_cols=feature_cols+['conn_type1','conn_type2','conn_type3']


# In[210]:

validation_check2(feature_cols,['c18_1','c18_2','c18_3'])


# In[211]:

feature_cols=feature_cols+['c18_1','c18_2','c18_3']


# In[212]:

print feature_cols


# In[213]:

t1.columns


# In[218]:

sorted_plot('weekday',t1,1)


# In[226]:

test['C15'].value_counts()


# In[228]:

train['C15'].value_counts()


# In[230]:

sorted_plot('C15',train,1)


# In[231]:

c15=train['C15'].unique()
for i in range(1,8,1):
    pair='c15_',format(i)
    str1=''.join(pair)
    t1[str1]=0
    t2[str1]=0
    test[str1]=0
    t1[str1][t1['C15']==c15[i-1]]=1
    t2[str1][t2['C15']==c15[i-1]]=1
    test[str1][test['C15']==c15[i-1]]=1


# In[233]:

l15=[]
for i in range(1,8,1):
    pair='c15_',format(i)
    str1=''.join(pair)
    l15=l15+[str1]
l15


# In[234]:

validation_check2(feature_cols,l15)


# In[235]:

feature_cols=feature_cols+l15


# In[236]:

print feature_cols


# In[237]:

test['C16'].value_counts()


# In[240]:

train['C16'].value_counts()


# In[241]:

c16=train['C16'].unique()
for i in range(1,9,1):
    pair='c16_',format(i)
    str1=''.join(pair)
    t1[str1]=0
    t2[str1]=0
    test[str1]=0
    t1[str1][t1['C16']==c16[i-1]]=1
    t2[str1][t2['C16']==c16[i-1]]=1
    test[str1][test['C16']==c16[i-1]]=1


# In[242]:

l=[]
for i in range(1,9,1):
    pair='c16_',format(i)
    str1=''.join(pair)
    l=l+[str1]


# In[243]:

validation_check2(feature_cols,l)


# In[244]:

feature_cols=feature_cols+l


# In[245]:

print feature_cols


# In[251]:

c21=train['C21'].value_counts()
c21=c21[c21>30]
c21=c21.index


# In[252]:

len(c21)


# In[253]:

avg_c21=np.array([])
for i in range(len(c21)):
    avg_c21=np.append(avg_c21,np.mean(train['click'][train['C21']==c21[i]]))


# In[254]:

plt.plot(avg_c21)


# In[255]:

df_c21=DataFrame({'c21':c21,'Avg':avg_c21})


# In[256]:

df_c21=df_c21.sort(columns='Avg')


# In[257]:

df_c21.head(2)


# In[258]:

plt.plot(df_c21['Avg'])


# In[259]:

print df_c21['Avg'].max(),df_c21['Avg'].min()


# In[261]:

df_c21.tail(2)


# In[262]:

t1['c21_high']=0
t2['c21_high']=0
test['c21_high']=0
t1['c21_high'][t1['C21']==35]=1
t2['c21_high'][t2['C21']==35]=1
test['c21_high'][test['C21']==35]=1


# In[263]:

validation_check2(feature_cols,['c21_high'])


# In[264]:

print feature_cols


# In[265]:

t1.columns


# In[267]:

t1['device_type'].value_counts()


# In[268]:

sorted_plot('device_type',train,1)


# In[269]:

np.mean(train['click'])


# In[270]:

t1['device_type_1']=0
t1['device_type_2']=0
t1['device_type_3']=0

t2['device_type_1']=0
t2['device_type_2']=0
t2['device_type_3']=0

test['device_type_1']=0
test['device_type_2']=0
test['device_type_3']=0

t1['device_type_1'][t1['device_type']==1]=1
t1['device_type_2'][t1['device_type']==2]=1
t1['device_type_3'][t1['device_type']==5]=1

t2['device_type_1'][t2['device_type']==1]=1
t2['device_type_2'][t2['device_type']==2]=1
t2['device_type_3'][t2['device_type']==5]=1

test['device_type_1'][test['device_type']==1]=1
test['device_type_2'][test['device_type']==2]=1
test['device_type_3'][test['device_type']==5]=1


# In[271]:

l=['device_type_1','device_type_2','device_type_3']


# In[272]:

validation_check2(feature_cols,l)


# In[274]:

feature_cols=feature_cols+l


# In[275]:

t1.columns


# In[279]:

train['app_category'].value_counts().shape


# In[280]:

t1['app_category'].value_counts().shape


# In[281]:

t1['app_category'].value_counts()


# In[282]:

app_cat=t1['app_category'].value_counts()
app_cat=app_cat[app_cat>30]
app_cat=app_cat.index
len(app_cat)


# In[284]:

avg_app_cat=np.array([])
for i in range(len(app_cat)):
    avg_app_cat=np.append(avg_app_cat,np.mean(train['click'][train['app_category']==app_cat[i]]))


# In[285]:

df_app_cat=DataFrame({'app_cat':app_cat,'avg':avg_app_cat})
df_app_cat=df_app_cat.sort(columns='avg')


# In[286]:

plt.plot(df_app_cat['avg'])


# In[287]:

plt.plot(df_app_cat['avg'], 'bo',df_app_cat['avg'], 'k')


# In[288]:

df_app_cat.max()


# In[289]:

t1['app_cat_high']=0
t2['app_cat_high']=0
test['app_cat_high']=0
t1['app_cat_high'][t1['app_category']=='fc6fa53d']=1
t2['app_cat_high'][t2['app_category']=='fc6fa53d']=1
test['app_cat_high'][test['app_category']=='fc6fa53d']=1


# In[292]:

validation_check2(feature_cols,['app_cat_high'])


# In[293]:

df_app_cat


# In[294]:

rank=[1,2,2,3,3,3,3,3,4,4,5,6,6,6,7,8]


# In[295]:

df_app_cat['rank']=rank


# In[296]:

df_app_cat


# In[297]:

for i in range(1,8,1):
    pair='app_cat_',format(i)
    str1=''.join(pair)
    t1[str1]=0
    t2[str1]=0
    test[str1]=0
    l=list(df_app_cat['app_cat'][df_app_cat['rank']==i])
    for k in l:
        t1[str1][t1['app_category']==k]=1
        t2[str1][t2['app_category']==k]=1
        test[str1][test['app_category']==k]=1


# In[298]:

l=[]
for i in range(1,8,1):
    pair='app_cat_',format(i)
    str1=''.join(pair)
    l=l+[str1]


# In[299]:

validation_check2(feature_cols,l)


# In[300]:

feature_cols=feature_cols+l


# In[301]:

print feature_cols


# In[311]:

sorted_plot('weekday',t1,2)


# In[312]:

print feature_cols


# In[313]:

t1.columns


# In[314]:

t1['site_id'].value_counts()


# In[ ]:



