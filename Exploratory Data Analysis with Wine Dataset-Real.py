#!/usr/bin/env python
# coding: utf-8

# # EXPLORATORY DATA ANALYSIS USING THE WINEQT DATASET
# 

# # BASIC EXPLORATION: This include
# - Importing the needed packages/libraries
# - Dataset size
# - Columns
# - Samples of rows
# - info
# - Dataset head and Tail
# - Describe ( To see the statistical summary)
# 
# 

# In[127]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[128]:


df = pd.read_csv('WineQT.csv')


# In[129]:


df.head()


# In[130]:


df.tail()


# In[131]:


df.info()


# In[132]:


df.shape


# In[133]:


df.describe()


# In[134]:


df.dtypes


# # CHECK THE NULL VALUES, DROP IRREVANT COLUMNS, CHECK DUPLICATE & REMOVE, UNIQUE VALUES

# In[135]:


df = df.drop("Id",axis=1)
df.head()


# In[136]:


df['quality'].unique()


# In[137]:


df.isnull().sum()


# In[138]:


duplicate = df.duplicated()
print(duplicate.sum())
df[duplicate]


# In[139]:


## Removing Duplicate
len_before = df.shape[0]
df.drop_duplicates(inplace=True)
len_after = df.shape[0]


print(f"before ={len_before}")
print(f"after = {len_after}")
print("")
print(f"Total remove= {len_before - len_after}")


# # OUTLIERS: DETECT AND REMOVE OUTLIER( HERE WE USED BOXPLOT AND INTER-QUANTILE RANGE TO CHECK THE OUTLIER. REMOVE THE OUTLIER WITH IQR SCORE)

# In[140]:


df.shape


# In[141]:


sns.boxplot(y=df['quality']), plt.show()


# In[142]:


sns.boxplot(y=df['fixed acidity']), plt.show()


# In[143]:


sns.boxplot(y=df['volatile acidity']), plt.show()


# In[144]:


sns.boxplot(y=df['citric acid']), plt.show()


# In[145]:


sns.boxplot(y=df['residual sugar']), plt.show()


# In[146]:


sns.boxplot(y=df['chlorides']), plt.show()


# In[147]:


sns.boxplot(y=df['free sulfur dioxide']), plt.show()


# In[148]:


sns.boxplot(y=df['total sulfur dioxide']), plt.show()


# In[149]:


sns.boxplot(y=df['density']), plt.show()


# In[150]:


sns.boxplot(y=df['pH']), plt.show()


# In[151]:


sns.boxplot(y=df['sulphates']), plt.show()


# In[152]:


sns.boxplot(y=df['alcohol']), plt.show()


# In[153]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[154]:


print(df<(Q1-1.5*IQR)) or (df>(Q3+1.5*IQR))


# # REMOVE OF OUTLIERS

# In[155]:


df_out= df[~((df<(Q1 - 1.5*IQR)) | (df>(Q3 + 1.5*IQR))).any(axis=1)]


# In[156]:


df_out.shape


# In[157]:


sns.boxplot(y=df_out['alcohol']), plt.show()


# In[158]:


sns.boxplot(y=df_out['residual sugar']), plt.show()


# In[159]:


sns.boxplot(y=df_out['quality']), plt.show()


# In[160]:


sns.boxplot(y=df_out['citric acid']), plt.show()


# In[161]:


sns.boxplot(y=df_out['chlorides']), plt.show()


# In[162]:


df_out.columns


# # VISUALIZING THE DATASET WITH THE FOLLOWING VISUALIZATIONS NAMELY ( Scatter_plot,Relplot,Line_plot,Heatmap,pair_plot )

# In[163]:


## Scatter Plots and 
sns.set_theme(style="darkgrid")
plt.figure(figsize=(16, 10))


# In[164]:


plt.figure(figsize=(16, 10))
sns.scatterplot(x="fixed acidity", y="volatile acidity", hue="quality", data=df_out);


# In[165]:


plt.figure(figsize=(16, 10))
sns.relplot(x="citric acid", y="residual sugar", hue="quality", data=df_out);


# In[166]:


plt.figure(figsize=(16, 10))
sns.relplot(x="chlorides", y="free sulfur dioxide", hue="quality", data=df_out);


# In[167]:


sns.relplot(x="total sulfur dioxide", y="density", hue="quality", data=df_out);


# In[168]:


sns.relplot(x="pH", y="sulphates", hue="quality", data=df_out);


# In[169]:


sns.relplot(x="alcohol", y="quality", data=df_out);


# In[170]:


### Line Plot
sns.lineplot(x="sulphates", y="quality", data=df_out)


# In[171]:


sns.lineplot(x="chlorides", y="quality", data=df_out)


# In[172]:


sns.lineplot(x="citric acid", y="quality", data=df_out)


# In[173]:


sns.lineplot(x="pH", y="quality", data=df_out)


# In[174]:


sns.lineplot(x="volatile acidity", y="quality", data=df_out)


# In[175]:


sns.lineplot(x="free sulfur dioxide", y="quality", data=df_out)


# In[176]:


sns.lineplot(x="total sulfur dioxide", y="quality", data=df_out)


# In[177]:


sns.lineplot(x="density", y="quality", data=df_out)


# In[178]:


sns.lineplot(x="fixed acidity", y="quality", data=df_out)


# In[179]:


sns.lineplot(x="residual sugar", y="quality", data=df_out)


# # Heatmap

# In[180]:


corr=df_out.corr()
f, ax= plt.subplots(figsize=(12,8))
sns.heatmap(corr, annot=True, square=False, ax=ax, linewidth=1)
plt.title('Pearson correlation of Features')


# # Pairsplots

# In[181]:



g=sns.pairplot(df_out, diag_kind = 'auto', hue = "quality")


# In[182]:


df_out.columns


# # Normalizing and Scaling

# In[183]:


from sklearn.preprocessing import StandardScaler
std_scale= StandardScaler()
std_scale


# In[184]:


df_out['fixed acidity'] =std_scale.fit_transform(df_out[['fixed acidity']])
df_out['volatile acidity'] =std_scale.fit_transform(df_out[['volatile acidity']])
df_out['citric acid'] =std_scale.fit_transform(df_out[['citric acid']])
df_out['residual sugar'] =std_scale.fit_transform(df_out[['residual sugar']])
df_out['chlorides'] =std_scale.fit_transform(df_out[['chlorides']])
df_out['free sulfur dioxide'] =std_scale.fit_transform(df_out[['free sulfur dioxide']])
df_out['total sulfur dioxide'] =std_scale.fit_transform(df_out[['total sulfur dioxide']])
df_out['density'] =std_scale.fit_transform(df_out[['density']])
df_out['pH'] =std_scale.fit_transform(df_out[['pH']])
df_out['sulphates'] =std_scale.fit_transform(df_out[['sulphates']])
df_out['alcohol'] =std_scale.fit_transform(df_out[['alcohol']])


# In[ ]:





# In[ ]:




