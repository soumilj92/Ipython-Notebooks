
# coding: utf-8

### Regular expressions in python 

# In[12]:

import re


# In[17]:

print (re.split(r'\s*', 'here are some words'))


# In[21]:

print (re.split(r'(\s*)', 'here   are     some words'))


# In[22]:

print (re.split(r'(s*)', 'here   are   some  some words'))


# In[23]:

print (re.split(r'[a-f]', 'lgfadylfblfrwfhwfljadljbf'))


# In[24]:

print (re.split(r'[a-f]', 'hfdF A kg' , re.I|re.M))


# In[25]:

print (re.split(r'[a-fA-F]', 'hfdF A kg' , re.I|re.M))


# In[28]:

import urllib
sites='google yahoo cnn msn'.split()

for s in sites:
    print ('Searching:' + s)
    try:
        u=urllib.urlopen('http://'+s+'.com')
    except:
        u=urllib.request.urlopen('http://'+s+'.com')
        
    text=u.read()
    title=re.findall(r'<title>+.*</title>+',str(text),re.I|re.M)
    print(title[0])


# In[31]:

print (re.split(r'[^A-Z]', 'hfdF A kg' , re.I|re.M))


# In[33]:

print (re.findall(r'[^A-Za-z0-9\s]', 'h+f=dF A @k-g!' , re.I|re.M))


# In[35]:

print (re.findall(r'[wW]oodchunks', 'woodchunks are great Woodchunks' , re.I|re.M))


# In[ ]:



