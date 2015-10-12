
# coding: utf-8

# In[1]:

import nltk


# In[3]:

nltk.download()


# In[34]:

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
from nltk import *


# In[5]:

os.chdir('desktop/insult')


# In[6]:

get_ipython().system(u'ls')


# In[7]:

test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')
ss=pd.read_csv('ss.csv')


# In[88]:

train.head(2)


##### We will divide the train texts into 2 texts - insults and non insults 

# In[115]:

insult=train[train['Insult']==1]
no_insult=train[train['Insult']==0]


# In[116]:

t_insult=list(insult['Comment'])
t_no_insult=list(no_insult['Comment'])


# In[117]:

insult=''.join(t_insult)
no_insult=''.join(t_no_insult)


# In[121]:

insult=str(insult)


# In[122]:

str3='""'
print str3


# In[123]:

insult=insult.strip('"')
insult


# In[214]:

insult=insult.replace('""',' ')
insult=insult.replace("\'","'")
insult=insult.replace('\\n'," ")
insult=insult.replace('\\xa0'," ")
insult=insult.replace('\\xc2'," ")
insult=insult.replace('\\u2028'," ")
insult=insult.replace('\\'," ")
insult=insult.replace('*'," ")
insult=insult.replace('%'," ")
insult=insult.replace('&'," ")
insult=insult.replace('#'," ")
insult=insult.replace('  '," ")
insult=insult.replace('!!',"!")
insult=insult.replace('..',".")
insult=insult.replace('--',"-")
insult=insult.replace('aaa',"a")
insult=insult.replace('AAA',"A")
insult=insult.replace('OOO',"O")
insult=insult.replace('rrr',"r")
insult=insult.replace('==',"=")
insult


# In[215]:

ins=insult.split(' ')


# In[216]:

len(ins)


# In[217]:

#finding long words gerater than size 15


# In[218]:

V = set(ins)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)


# In[219]:

ins


# In[240]:

ins1=FreqDist(ins)


# In[237]:

no_insult=no_insult.replace('""',' ')
no_insult=no_insult.replace("\'","'")
no_insult=no_insult.replace('\\n'," ")
no_insult=no_insult.replace('\\xa0'," ")
no_insult=no_insult.replace('\\xc2'," ")
no_insult=no_insult.replace('\\u2028'," ")
no_insult=no_insult.replace('\\'," ")
no_insult=no_insult.replace('*'," ")
no_insult=no_insult.replace('%'," ")
no_insult=no_insult.replace('&'," ")
no_insult=no_insult.replace('#'," ")
no_insult=no_insult.replace('  '," ")
no_insult=no_insult.replace('!!',"!")
no_insult=no_insult.replace('..',".")
no_insult=no_insult.replace('--',"-")
no_insult=no_insult.replace('aaa',"a")
no_insult=no_insult.replace('AAA',"A")
no_insult=no_insult.replace('OOO',"O")
no_insult=no_insult.replace('rrr',"r")
no_insult=no_insult.replace('==',"=")
no_insult=no_insult.replace('u0111'," ")
no_insult=no_insult.replace('xf3'," ")
no_insult=no_insult.replace('xf4n'," ")
no_insult=no_insult.replace('u01b0'," ")
no_insult=no_insult.replace('xe1c'," ")
no_insult=no_insult.replace('_'," ")
no_insult=no_insult.replace('u0103n'," ")
no_insult=no_insult.replace('xf2n'," ")
no_insult=no_insult.replace('u0169ng'," ")
no_insult=no_insult.replace('u0103n'," ")
no_insult=no_insult.replace('u1ec9'," ")
no_insult=no_insult.replace('xec'," ")
no_insult=no_insult.replace('u1ee3c'," ")
no_insult=no_insult.replace('u1ed1t'," ")
no_insult=no_insult.replace('xe2n'," ")
no_insult=no_insult.replace('u1ed3i'," ")
no_insult=no_insult.replace('u1ed1t'," ")
no_insult=no_insult.replace('cu'," ")
no_insult=no_insult.replace('u1ee5'," ")
no_insult=no_insult.replace('c'," ")
no_insult=no_insult.replace('C'," ")
no_insult=no_insult.replace('b'," ")
no_insult=no_insult.replace('y'," ")
no_insult=no_insult.replace('xeau'," ")
no_insult=no_insult.replace('g'," ")
no_insult=no_insult.replace('kh'," ")
no_insult


# In[239]:

nins=no_insult.split(' ')


# In[241]:

nins1=FreqDist(nins)


# In[244]:

type(ins1)


# In[245]:

ins1


# In[254]:

ins1=dict(ins1)
type(ins1)


# In[261]:

ins1['fuck']


# In[262]:

len(ins1)


# In[265]:

ins1=DataFrame(ins1.items(),columns=['word','count'])


# In[268]:

ins1=ins1.sort(columns='count')


# In[270]:

ins1=ins1[ins1['count']>5]


# In[271]:

ins1.shape


# In[275]:

ins1.head(2)


# In[279]:

ins1.index=range(len(ins1))


# In[281]:

ins1['word'][1]


# In[285]:

ins1['pos']=''
for i in range(len(ins1)):
    ins1['pos'][i]= nltk.pos_tag(str(ins1['word'][i]))


# In[286]:

ins1.head(2)


# In[289]:

test['words']=''


# In[290]:

test.head(2)


# In[293]:

for i in range(len(test)):
    test['Comment'][i]=test['Comment'][i].replace('.','')
    test['words'][i]=test['Comment'][i].split(' ')


# In[294]:

test.head(2)


# In[295]:

ss.head(1)


# In[296]:

test['Insult']=0


# In[297]:

test.head(2)


# In[298]:

ins1.head(2)


# In[299]:

ins1.shape


# In[300]:

ins1['len']=0
for i in range(len(ins1)):
    ins1['len'][i]=len(ins1['word'][i])


# In[301]:

ins1.head(3)


# In[302]:

ins1=ins1[ins1['len']>2]


# In[303]:

ins1.shape


# In[305]:

ins1.index=range(len(ins1))


# In[306]:

ins1.head(3)


# In[307]:

ins1['word'][0]


# In[308]:

ins1[ins1['word']=='fuck']


# In[309]:

test.head(2)


# In[381]:

for i in range(len(test)):
    for j in test['words'][i]:
        if j.find('fuck')>=0:
            test['Insult'][i]=1
            break
        if j.find('ass')>=0:
            test['Insult'][i]=1
            break
        if j.find('pussy')>=0:
            test['Insult'][i]=1
            break
        if j.find('retard')>=0:
            test['Insult'][i]=1
            break
        if j.find('balls')>=0:
            test['Insult'][i]=1
            break
        if j.find('troll')>=0:
            test['Insult'][i]=1
            break
        if j.find('you?')>=0:
            test['Insult'][i]=1
            break
        if j.find('racist')>=0:
            test['Insult'][i]=1
            break
        if j.find('crap')>=0:
            test['Insult'][i]=1
            break
        if j.find('bitch')>=0:
            test['Insult'][i]=1
            break
        if j.find('ugly')>=0:
            test['Insult'][i]=1
            break
        if j.find('dead')>=0:
            test['Insult'][i]=1
            break
        if j.find('fool')>=0:
            test['Insult'][i]=1
            break
        if j.find('loser')>=0:
            test['Insult'][i]=1
            break
        if j.find('idiot')>=0:
            test['Insult'][i]=1
            break
        if j.find('hell')>=0:
            test['Insult'][i]=1
            break
        if j.find('stupid')>=0:
            test['Insult'][i]=1
            break
        if j.find('moron')>=0:
            test['Insult'][i]=1
            break
        if j.find('suck')>=0:
            test['Insult'][i]=1
            break
        if j.find('gay')>=0:
            test['Insult'][i]=1
            break
        if j.find('nigga')>=0:
            test['Insult'][i]=1
            break
        if j.find('dick')>=0:
            test['Insult'][i]=1
            break
        if j.find('shit')>=0:
            test['Insult'][i]=1
            break


# In[382]:

sum(test['Insult'])


# In[383]:

len(test['Insult'])


# In[384]:

ss['Insult']=test['Insult']


# In[385]:

ss.to_csv('submit3.csv',index=False)


# In[315]:

ins1


# In[380]:

ins1['word'][296:]


# In[ ]:



