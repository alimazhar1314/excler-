#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import seaborn as sns
import statsmodels.api as sma


# # step 1
# import files

# In[9]:


df=pd.read_csv("ToyotaCorolla.csv",encoding="latin1")
df


# sort the data

# In[10]:


dfn = pd.concat([df.iloc[:,2:4],df.iloc[:,6:7],df.iloc[:,8:9],df.iloc[:,12:14],df.iloc[:,15:18]],axis=1)
dfn


# Renaming the data

# In[11]:


dfnw = dfn.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
dfnw


# In[12]:


dfnew = dfnw.drop_duplicates().reset_index(drop=True)
dfnew
dfnew.describe()


# Correltion

# In[13]:


dfnew.corr()


# # step 2
# split the variables in x & y

# In[14]:


X = dfnew[["Age"]]


# In[15]:


X = dfnew[["Age","Weight"]]


# In[16]:


X = dfnew[["Age","Weight","KM"]]


# In[17]:


X = dfnew[["Age","Weight","KM","HP"]]


# In[18]:


X = dfnew[["Age","Weight","KM","HP","QT"]]


# In[19]:


X = dfnew[["Age","Weight","KM","HP","QT","Doors"]]


# In[20]:


X = dfnew[["Age","Weight","KM","HP","QT","Doors","CC"]]


# In[21]:


X = dfnew[["Age","Weight","KM","HP","QT","Doors","CC","Gears"]]


# In[ ]:





# In[22]:


Y = dfnew["Price"]


# scatter plot

# In[25]:


sns.set_style(style='darkgrid')
sns.pairplot(dfnew)


# In[28]:


X_new = sma.add_constant(X)
lmreg = sma.OLS(Y,X_new).fit()
lmreg.summary()


# # step 3

# residual analysis

# In[30]:


import matplotlib.pyplot as plt
import statsmodels.api as sm
qqplot=sm.qqplot(lmreg.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[31]:


lmreg.resid.hist()
lmreg.resid
list(np.where(lmreg.resid>10))


# # step 4

# Model validation

# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test,Y_train, Y_test = train_test_split(X,Y , test_size=0.3,random_state=(42))


# # step 5

# Model fitting

# In[37]:


from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train,Y_train)
Y_pred_train = LR.predict(X_train)
Y_pred_test = LR.predict(X_test)


# # step 6
# metrics

# In[38]:


from sklearn.metrics import mean_squared_error
mse1= mean_squared_error(Y_train,Y_pred_train)
RMSE1 = np.sqrt(mse1)
print("Training error: ", RMSE1.round(2))


# In[39]:


mse2= mean_squared_error(Y_test,Y_pred_test)
RMSE2 = np.sqrt(mse2)
print("Test error: ", RMSE2.round(2))


# # step 7

# cook's distance

# In[40]:


lmreg_influence = lmreg.get_influence()
(cooks, pvalue) = lmreg_influence.cooks_distance


# In[41]:


cooks = pd.DataFrame(cooks)
cooks[0].describe()


# In[ ]:




