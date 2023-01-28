#!/usr/bin/env python
# coding: utf-8

# OLUYINKA OGUNDIPE (The First Data Bender)

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# In[27]:


pip install dmba


# In[28]:


from dmba import regressionSummary


# In[2]:


housing_df = pd.read_csv('WestRoxbury.csv')


# In[3]:


housing_df.describe()


# In[4]:


#this replace the space between the columns name with underscore '_'
housing_df.columns = [s.strip().replace(' ', '_') 
   for s in housing_df.columns]


# In[5]:


housing_df.describe()


# In[31]:


#Illustrate missing data procedure to enable you work with your data
# convert some rows in variable Bedroom to NA then input missing values using the median 
missingRows=housing_df.sample(10).index
housing_df.loc[missingRows,'BEDROOMS']=np.nan
print('Number of rows with valid BEDROOMS values after setting to NAN; ',housing_df['BEDROOMS'].count())


# In[32]:


#remove rows with missing values
reduced_df=housing_df.dropna()
print('Number of rows after removing rows with missing values: ', len(reduced_df))


# In[33]:


#replace the missing values using the median of the remaining values
medianBedrooms=housing_df['BEDROOMS'].median()
housing_df.BEDROOMS=housing_df.BEDROOMS.fillna(value=medianBedrooms)
print('Number of rows with valid BEDROOMS values after filling NA values: ', housing_df['BEDROOMS'].count())


# In[6]:


housing_df.loc[0:3]  


# In[7]:


housing_df['TOTAL_VALUE'].iloc[0:10]


# In[8]:


housing_df.iloc[:,0:1]


# In[9]:


housing_df.sample(5)


# In[10]:


# We use this here to oversample houses with over 10 rooms
weights = [0.9 if rooms > 10 else 0.01 for rooms in housing_df.ROOMS]
housing_df.sample(5, weights=weights)


# In[11]:


housing_df.dtypes


# In[12]:


housing_df.loc[0:10,'REMODEL']


# In[13]:


housing_df.REMODEL.unique()


# In[14]:


print(housing_df.REMODEL.dtype)


# In[15]:


#Remodel is converted to categorical variable
housing_df.REMODEL = housing_df.REMODEL.astype('category')


# In[16]:


print(housing_df.REMODEL.dtype)


# In[17]:


#create binary dummy & drop one variable to prevent redundancy
housing_df = pd.get_dummies(housing_df, prefix_sep='_', drop_first=True)


# In[18]:


housing_df.columns


# In[19]:


print(housing_df.loc[:, 'REMODEL_Old':'REMODEL_Recent'].head(5))


# In[20]:


excludeColumns = ('TOTAL_VALUE', 'TAX')
predictors = [s for s in housing_df.columns if s 
   not in excludeColumns]
outcome = 'TOTAL_VALUE'


# In[21]:


print(housing_df[predictors])


# In[22]:


#
X = housing_df[predictors]
y = housing_df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)


# In[23]:


model = LinearRegression()
model.fit(train_X, train_y)

train_pred = model.predict(train_X)
train_results = pd.DataFrame({
    'TOTAL_VALUE': train_y, 
    'predicted': train_pred, 
    'residual': train_y - train_pred
})


# In[24]:


train_results.head()


# In[25]:


valid_pred = model.predict(valid_X)
valid_results = pd.DataFrame({
    'TOTAL_VALUE': valid_y, 
    'predicted': valid_pred, 
    'residual': valid_y - valid_pred
})


# In[26]:


valid_results.head()


# In[29]:


regressionSummary(train_results.TOTAL_VALUE, 
   train_results.predicted)


# In[30]:


regressionSummary(valid_results.TOTAL_VALUE, valid_results.predicted)

