#!/usr/bin/env python
# coding: utf-8

# # PREDICTION OF HOUSE PRICE UNIT AREA USING LINEAR REGRESSION MODEL WITH REAL ESTATE DATASET

# # IMPORT LIBRARIES

# In[27]:


import pandas as pd
import numpy as np


# # LOADING OF DATA

# In[28]:


df = pd.read_csv('Real estate.csv')
df.head()


# In[29]:


df.drop('No', inplace=True, axis=1)


# In[40]:


df.isnull().sum()


# In[30]:


df.head()


# # DEFINING X AND Y 

# In[31]:


x = df.drop(['Y house price of unit area'], axis=1).values
y = df["Y house price of unit area"].values


# In[32]:


print(x)


# In[33]:


print(y)


# # SPLITS DATASET INTO TRAINING & TESTING

# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# # TRAIN THE MODEL ON TRAINING DATASET

# In[35]:


from sklearn.linear_model import LinearRegression
Reg = LinearRegression()
Reg.fit(x_train,y_train)


# # PREDICT MODEL

# In[36]:


y_predicted = Reg.predict(x_test)
print(y_predicted)


# In[37]:


Reg.predict([[2012.917,32.0,84.87882,10,24.98298,121.54024]])


# # EVALUATE THE MODEL

# In[38]:


from sklearn.metrics import r2_score
r2_score(y_test,y_predicted)


# # PLOT THE RESULTS

# In[39]:


from matplotlib import pyplot as plt
plt.figure(figsize = (15,10))
plt.scatter(y_test,y_predicted)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')


# # PREDICTED VALUES

# In[26]:


pred_y_df = pd.DataFrame({'Actual values':y_test,'Predict':y_predicted, 'Difference': y_test-y_predicted})
pred_y_df[0:20]


# In[ ]:




