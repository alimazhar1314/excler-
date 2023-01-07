#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # step 1
# import files

# In[2]:


df=pd.read_csv("50_Startups.csv")
df


# Correltion

# In[3]:


df.corr()


# Split the variabes in X & Y

# In[4]:


X = df[["R&D Spend"]] # R2: 0.947, RMSE: 9226.101


# In[5]:


X = df[["R&D Spend","Administration"]] # R2: 0.948, RMSE: 9115.198


# In[6]:


X = df[["Marketing Spend"]] # R2: 0.559, RMSE: 26492.829


# In[7]:


X = df[["Marketing Spend","Administration"]] # R2: 0.610, RMSE: 24927.067


# In[8]:


X = df[["R&D Spend","Marketing Spend","Administration"]] # R2: 0.951, RMSE: 8855.344


# In[9]:


Y = df["Profit"]


# scatter plot 

# In[10]:


import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)


# In[11]:


import statsmodels.api as sma

X_new = sma.add_constant(X)
lmreg = sma.OLS(Y,X_new).fit()
lmreg.summary()


# Residual analysis

# In[12]:


import matplotlib.pyplot as plt
import statsmodels.api as sm
qqplot=sm.qqplot(lmreg.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[13]:


lmreg.resid.hist()
lmreg.resid
list(np.where(lmreg.resid>10))


# Model validation

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X,Y , test_size=0.3,random_state=(42))


# # step 2

# Model fitting

# In[15]:


from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train,Y_train)
Y_pred_train = LR.predict(X_train)
Y_pred_test = LR.predict(X_test)


# # step 3
# metrics

# In[16]:


from sklearn.metrics import mean_squared_error
mse1= mean_squared_error(Y_train,Y_pred_train)
RMSE1 = np.sqrt(mse1)
print("Training error: ", RMSE1.round(2))


# In[17]:


mse2= mean_squared_error(Y_test,Y_pred_test)
RMSE2 = np.sqrt(mse2)
print("Test error: ", RMSE2.round(2))


# Model validation 

# In[21]:


Training_error = []
Test_error = []


# In[23]:


for i in range(1,500):
    X_train, X_test,Y_train, Y_test = train_test_split(X,Y , test_size=0.3,random_state=(i))
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    Training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)).round(2))
    Test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)).round(2))


# In[24]:


print(Training_error)
print(Test_error)


# In[25]:


print("validationset approach for Traning: ",np.mean(Training_error).round(2))    
print("validationset approach for test: ",np.mean(Test_error).round(2))


# # step 4
# model deletion

# In[26]:


lmreg_influence = lmreg.get_influence()
(cooks, pvalue) = lmreg_influence.cooks_distance


# Cook's distance

# In[27]:


cooks = pd.DataFrame(cooks)
cooks[0].describe()


# In[ ]:




