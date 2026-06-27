#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import essential data manipulation and analysis libraries
import pandas as pd  # For data manipulation and analysis with DataFrames
import numpy as np   # For numerical computing and array operations

# Import visualization libraries
import matplotlib.pyplot as plt  # For creating static plots and charts
import seaborn as sns           # For statistical data visualization

# Import machine learning utilities from scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.preprocessing import StandardScaler      # For feature scaling and normalization
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,classification_report  # For model evaluation metrics


# In[2]:


# Load the Novagen dataset from CSV file into a pandas DataFrame
df = pd.read_csv("novagen_dataset.csv")


# In[3]:


# Display the first 5 rows of the DataFrame to examine the data structure and content
df.head()


# In[4]:


# Get the dimensions of the DataFrame (rows, columns)
df.shape


# In[5]:


# Display the column names of the DataFrame
df.columns


# In[6]:


# Display comprehensive information about the DataFrame including data types, non-null counts, and memory usage
df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


df.nunique()


# In[11]:


sns.countplot(x='Target',data=df)


# In[12]:


df.hist(figsize=(15,10))


# In[13]:


sns.heatmap(df.corr(),annot=True)


# In[14]:


sns.boxplot(df["BMI"])


# In[15]:


from sklearn.impute import SimpleImputer
SimpleImputer()


# In[16]:


X=df.drop("Target",axis=1)

y=df["Target"]


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.2,random_state=42,stratify=y
)


# In[18]:


scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)


# In[19]:


from xgboost import XGBClassifier

model = XGBClassifier(random_state=42)

model.fit(X_train,y_train)


# In[20]:


y_pred = model.predict(X_test)
print("Accuracy Score",accuracy_score(y_pred,y_test))


# In[21]:


print("Precision Score",precision_score(y_pred,y_test))


# In[44]:


print("F1 Score",f1_score(y_pred,y_test))


# In[46]:


print("Classification Report:",classification_report(y_pred,y_test))


# In[ ]:




