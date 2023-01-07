#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageGrab
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm


# # Question 1

# Import files

# In[3]:


cutlets = pd.read_csv('Cutlets.csv')
cutlets.head(10)


# In[4]:


cutlets.describe()


# Check for null values.

# In[5]:


cutlets.isnull().sum()


# Check for duplicate values.

# In[6]:


cutlets[cutlets.duplicated()].shape


# Check for data types.

# In[7]:


cutlets.info()


# Plot the data.

# In[8]:


plt.subplots(figsize = (9,6))
plt.subplot(121)
plt.boxplot(cutlets['Unit A'])
plt.title('Unit A')
plt.subplot(122)
plt.boxplot(cutlets['Unit B'])
plt.title('Unit B')
plt.show()


# Plot Histogram.

# In[9]:


plt.subplots(figsize = (9,6))
plt.subplot(121)
plt.hist(cutlets['Unit A'], bins = 15)
plt.title('Unit A')
plt.subplot(122)
plt.hist(cutlets['Unit B'], bins = 15)
plt.title('Unit B')
plt.show()


# Distribution plot.

# In[10]:


plt.figure(figsize = (8,6))
labels = ['Unit A', 'Unit B']
sns.distplot(cutlets['Unit A'], kde = True)
sns.distplot(cutlets['Unit B'],hist = True)
plt.legend(labels)


# Q Q plot.

# In[11]:


sm.qqplot(cutlets["Unit A"], line = 'q')
plt.title('Unit A')
sm.qqplot(cutlets["Unit B"], line = 'q')
plt.title('Unit B')
plt.show()


# Compare evidenve with hypothesis using t-statistics.

# In[12]:


statistic , p_value = stats.ttest_ind(cutlets['Unit A'],cutlets['Unit B'], alternative = 'two-sided')
print('p_value=',p_value)


# Interpretate p value.

# In[13]:


alpha = 0.025
print('Significnace=%.3f, p=%.3f' % (alpha, p_value))
if p_value <= alpha:
    print('We reject Null Hypothesis there is a significance difference between two Units A and B')
else:
    print('We fail to reject Null hypothesis')


# # Question 2

# Import files

# In[17]:


labtat=pd.read_csv('LabTAT.csv')
labtat.head()


# Check for null values.

# In[18]:


labtat.isnull().sum()


# Compare Evidences with Hypothesis using t-statictic

# In[19]:


test_statistic , p_value = stats.f_oneway(labtat.iloc[:,0],labtat.iloc[:,1],labtat.iloc[:,2],labtat.iloc[:,3])
print('p_value =',p_value)


# In[20]:


alpha = 0.05
print('Significnace=%.3f, p=%.3f' % (alpha, p_value))
if p_value <= alpha:
    print('We reject Null Hypothesis there is a significance difference between TAT of reports of the laboratories')
else:
    print('We fail to reject Null hypothesis')


# # Question 3

# Import files

# In[22]:


buyer=pd.read_csv('BuyerRatio.csv')
buyer.head()


# Apply chisquare test

# In[27]:


stats.chi2_contingency(table)


# Compare evidence wit hyopthesis

# # Question 4

# Import files

# In[32]:


customer=pd.read_csv('Costomer+OrderForm.csv')
customer.head()


# Check for null values

# In[35]:


customer.isnull().sum()


# Data types

# In[36]:


customer.info()


# Checking value counts in data

# Creating Contingency table

# In[40]:


contingency_table = [[271,267,269,280],
                    [29,33,31,20]]
print(contingency_table)


# Calculating Expected Values for Observed data

# In[41]:


stat, p, df, exp = stats.chi2_contingency(contingency_table)
print("Statistics = ",stat,"\n",'P_Value = ', p,'\n', 'degree of freedom =', df,'\n', 'Expected Values = ', exp)


# Defining Expected values and observed values

# In[42]:


observed = np.array([271, 267, 269, 280, 29, 33, 31, 20])
expected = np.array([271.75, 271.75, 271.75, 271.75, 28.25, 28.25, 28.25, 28.25])


# Compare Evidences with Hypothesis using t-statictic

# In[43]:


test_statistic , p_value = stats.chisquare(observed, expected, ddof = df)
print("Test Statistic = ",test_statistic,'\n', 'p_value =',p_value)


# Interprtate p value

# In[44]:


alpha = 0.05
print('Significnace=%.3f, p=%.3f' % (alpha, p_value))
if p_value <= alpha:
    print('We reject Null Hypothesis there is a significance difference between TAT of reports of the laboratories')
else:
    print('We fail to reject Null hypothesis')


# In[ ]:





# In[ ]:




