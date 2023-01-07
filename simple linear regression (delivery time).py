#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as smf
import statsmodels.formula.api as sm
import warnings
warnings.filterwarnings('ignore')


# # step 1
# import files

# In[2]:


df = pd.read_csv('delivery_time.csv')
df


# # step 2

# Performiing EDA on data

# In[3]:


df1 = df.rename({'Delivery Time':'Delivery_Time','Sorting Time':'Sorting_Time'}, axis = 1)
df1


# Check for datatypes

# In[4]:



df.info()


# Check for null values

# In[5]:


df.isnull().sum()


# Check for duplicate values

# In[6]:


df[df.duplicated()].shape


# # step 3

# Plot the data and check for outliers

# In[7]:


plt.subplots(figsize = (9,6))
plt.subplot(121)
plt.boxplot(df['Delivery Time'])
plt.title('Delivery Time')
plt.subplot(122)
plt.boxplot(df['Sorting Time'])
plt.title('Sorting Time')
plt.show()


# # step 4

# Check for correlation between variables 

# In[8]:


df.corr()


# Visualize data 

# In[9]:


sns.regplot(x=df['Sorting Time'],y=df['Delivery Time'])  


# In[10]:


df.var()


# # step 5

# In[ ]:


Feature engineering


# In[11]:


sns.distplot(df['Delivery Time'], bins = 10, kde = True)
plt.title('Before Transformation')
sns.displot(np.log(df['Delivery Time']), bins = 10, kde = True)
plt.title('After Transformation')
plt.show()


# In[12]:


labels = ['Before Transformation','After Transformation']
sns.distplot(df['Delivery Time'], bins = 10, kde = True)
sns.distplot(np.log(df['Delivery Time']), bins = 10, kde = True)
plt.legend(labels)
plt.show()


# In[13]:



smf.qqplot(df['Delivery Time'], line = 'r')
plt.title('No transformation')
smf.qqplot(np.log(df['Delivery Time']), line = 'r')
plt.title('Log transformation')
smf.qqplot(np.sqrt(df['Delivery Time']), line = 'r')
plt.title('Square root transformation')
smf.qqplot(np.cbrt(df['Delivery Time']), line = 'r')
plt.title('Cube root transformation')
plt.show()


# In[14]:


labels = ['Before Transformation','After Transformation']
sns.distplot(df['Sorting Time'], bins = 10, kde = True)
sns.distplot(np.log(df['Sorting Time']), bins = 10, kde = True)
plt.legend(labels)
plt.show()


# In[15]:


smf.qqplot(df['Sorting Time'], line = 'r')
plt.title('No transformation')
smf.qqplot(np.log(df['Sorting Time']), line = 'r')
plt.title('Log transformation')
smf.qqplot(np.sqrt(df['Sorting Time']), line = 'r')
plt.title('square root transformation')
smf.qqplot(np.cbrt(df['Sorting Time']), line = 'r')
plt.title('Cube root transformation')
plt.show()


# # step 6

# Fit the linear regression model

# In[16]:


model = sm.ols('Delivery_Time~Sorting_Time', data = df1).fit()


# In[17]:


model.summary()


# Square root transformtaion on data

# In[18]:


model1 = sm.ols('np.sqrt(Delivery_Time)~np.sqrt(Sorting_Time)', data = df1).fit()
model1.summary()


# Cube root transformation on data

# In[19]:


model2 = sm.ols('np.cbrt(Delivery_Time)~np.cbrt(Sorting_Time)', data = df1).fit()
model2.summary()


# Log transformation on data

# In[20]:


model3 = sm.ols('np.log(Delivery_Time)~np.log(Sorting_Time)', data = df1).fit()
model3.summary()


# Model testing

# In[21]:


model.params


# In[22]:


print(model.tvalues,'\n',model.pvalues)


# In[23]:


model.rsquared,model.rsquared_adj


# # step 7

# Residual analysis

# In[24]:


import statsmodels.api as sm
sm.qqplot(model.resid, line = 'q')
plt.title('Normal Q-Q plot of residuals of Model without any data transformation')
plt.show()


# In[25]:


sm.qqplot(model2.resid, line = 'q')
plt.title('Normal Q-Q plot of residuals of Model with Log transformation')
plt.show()


# # step 8

# Model validation

# In[26]:


from sklearn.metrics import mean_squared_error


# In[27]:


model1_pred_y =np.square(model1.predict(df1['Sorting_Time']))
model2_pred_y =pow(model2.predict(df1['Sorting_Time']),3)
model3_pred_y =np.exp(model3.predict(df1['Sorting_Time']))


# # step 9

# Predicting values from model with log transformation

# In[30]:


predicted = pd.DataFrame()
predicted['Sorting_Time'] = df1.Sorting_Time
predicted['Delivery_Time'] = df1.Delivery_Time
predicted['Predicted_Delivery_Time'] = pd.DataFrame(np.exp(model2.predict(predicted.Sorting_Time)))
predicted


# Predicting original data

# In[31]:


predicted1 = pd.DataFrame()
predicted1['Sorting_Time'] = df1.Sorting_Time
predicted1['Delivery_Time'] = df1.Delivery_Time
predicted1['Predicted_Delivery_Time'] = pd.DataFrame(model.predict(predicted1.Sorting_Time))
predicted1


# In[ ]:





# In[ ]:





# In[ ]:




