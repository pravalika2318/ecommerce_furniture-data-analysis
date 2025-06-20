

# In[2]:


from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np
import pandas as pd
df = pd.read_csv(
    r"C:\Users\prava\Downloads\ecommerce_furniture_dataset_2024.csv")
df.head()


# In[4]:


df.isnull().sum()


# In[6]:


df.shape


# In[8]:


df.drop(['originalPrice'], axis=1, inplace=True)


# In[10]:


df.head()


# In[12]:


df['tagText'].nunique()


# In[14]:


df['tagText'].value_counts()


# In[22]:


df['tagText'] = df['tagText'].apply(
    lambda x: x if x in ['Free shipping', '+Shipping: $5.09'] else 'others')
print(df['tagText'].value_counts())


# In[26]:


sns.countplot(x='tagText', data=df)


# In[28]:


df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df.head()


# In[30]:


sns.distplot(df['price'])


# In[32]:


sns.distplot(df['sold'])


# In[34]:


sns.scatterplot(x='price', y='sold', data=df)


# In[36]:


filtered_df = df[df['tagText'] == 'Free shipping']
sns.pairplot(filtered_df[['price', 'sold']])


# In[38]:


le = LabelEncoder()
df['tagText'] = le.fit_transform(df['tagText'])
df.head()


# In[40]:


df['tagText'].value_counts()


# In[ ]:
