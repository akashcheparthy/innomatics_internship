#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from the given dataset i found the linear relation between the two columns provided
#i created a model using linear regression function


# In[1]:


import pandas


# In[4]:


dataset=pandas.read_csv('data_2_var.csv')


# In[5]:


dataset


# In[6]:


y=dataset['B']


# In[7]:


y


# In[8]:


x=dataset['A']


# In[13]:


x=x.values.reshape(-1,1)


# In[15]:


x


# In[ ]:





# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


model= LinearRegression()


# In[14]:


model.fit(x,y)


# In[16]:


model


# In[17]:


model.predict([[-113.367]])


# In[19]:


weight=model.coef_


# In[20]:


weight


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




