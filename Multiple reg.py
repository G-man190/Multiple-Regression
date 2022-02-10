#!/usr/bin/env python
# coding: utf-8

# # Loading Libraries

# In[1]:


import numpy as np
import pandas as pd


# # Loading Dataset

# In[2]:


df = pd.read_excel('CombineML2.xlsx')
df.head()


# In[3]:


df.shape


# # Defining x and y

# In[4]:


x = df.drop(['PE'],axis = 1).values
y = df['PE'].values


# # Spliting the dataset

# In[5]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3,random_state=0)


# # Modelling 

# In[19]:


from sklearn.linear_model import LinearRegression
Reg = LinearRegression()
model =Reg.fit(x_train,y_train)


# # Prediction Model

# In[23]:


y_predicted = model.predict(x_test)
print(y_predicted)


# # Evalute model

# In[24]:


from sklearn.metrics import r2_score
r2_score(y_test,y_predicted)


# In[ ]:





# # Ploting 

# In[27]:


from matplotlib import pyplot as plt

plt.figure(figsize= (10,15))
plt.scatter(y_test,y_predicted)
plt.xlabel('Actual')
plt.ylabel('Predict')
plt.title('Actual vs Predict')


# In[18]:


New_Data=pd.DataFrame({'Actual Data':y_test,'Predict': y_predicted, 'Difference': y_test - y_predicted})
New_Data[0:20]


# In[ ]:




