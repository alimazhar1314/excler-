#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


df=pd.read_csv("Salary_Data.csv")
df


# spilt the variable

# In[8]:


X=df[["YearsExperience"]]
X
Y=df["Salary"]
Y


# EDA (Scatter plot)

# In[9]:



plt.scatter(X.iloc[:,0],Y,color='red')
plt.ylabel("Salary")
plt.xlabel("YearsExperience")
plt.show()


# Model fitting

# In[10]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y2=LR.predict(X)
Y2


# Calculate RMSE,R square

# In[11]:


from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
mse=mean_squared_error(Y,Y2)
RMSE=np.sqrt(mse)
print("Root mean square value:",RMSE)
print("R square value:",r2_score(Y,Y2))


# log transformation

# In[12]:


X=df[["YearsExperience"]]
X
Y=np.log(df["Salary"])
Y


# EDA (Scatter plot)

# In[13]:



plt.scatter(X.iloc[:,0],Y,color='red')
plt.ylabel("Salary")
plt.xlabel("YearsExperience")
plt.show()


# Model fitting

# In[14]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y2=LR.predict(X)
Y2


# Calculate RMSE,R square

# In[15]:


from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
mse=mean_squared_error(Y,Y2)
RMSE=np.sqrt(mse)
print("Root mean square value:",RMSE)
print("R square value:",r2_score(Y,Y2))


# square root transformations 

# In[16]:


X=df[["YearsExperience"]]
X
Y=np.sqrt(df["Salary"])
Y


#  EDA (Scatter plot)

# In[17]:


plt.scatter(X.iloc[:,0],Y,color='red')
plt.ylabel("Salary")
plt.xlabel("YearsExperience")
plt.show()


# In[18]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y2=LR.predict(X)
Y2


# Calculate RMSE,R square

# In[19]:


mse=mean_squared_error(Y,Y2)
RMSE=np.sqrt(mse)
print("Root mean square value:",RMSE)
print("R square value:",r2_score(Y,Y2))


# In[ ]:




