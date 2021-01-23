#!/usr/bin/env python
# coding: utf-8

# In[193]:


import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[194]:


df = pd.read_csv("data.csv")

#print(df.head())
x = df.drop("HighSalary",axis = 1)
y = df.HighSalary
#x


# In[195]:


df.columns


# In[196]:


df = df[df.board10 != 0]


# In[197]:


df = df[df.board12 != 0]


# In[198]:


df.dtypes


# In[199]:



dic_gender={j:i for i,j in enumerate(list(df.Gender.unique()))}
dic_dob={j:i for i,j in enumerate(list(df.DOB.unique()))}
dic_10board={j:i for i,j in enumerate(list(df.board10.unique()))}
dic_12board={j:i for i,j in enumerate(list(df.board12.unique()))}
dic_spec={j:i for i,j in enumerate(list(df.Specialization.unique()))}
dic_degree={j:i for i,j in enumerate(list(df.Degree.unique()))}
dic_collegestate={j:i for i,j in enumerate(list(df.CollegeState.unique()))}


# In[200]:


def number_spec(row):
  text = row['Specialization']
  return dic_spec[text]
df['Specialization'] = df.apply(number_spec,axis=1)


# In[201]:


def number_gender(row):
  text = row['Gender']
  return dic_gender[text]
df['Gender'] = df.apply(number_gender,axis=1)


# In[202]:


def number_dob(row):
  text = row['DOB']
  return dic_dob[text]
df['DOB'] = df.apply(number_dob,axis=1)


# In[203]:


def number_10board(row):
  text = row['board10']
  return dic_10board[text]
df['10board'] = df.apply(number_10board,axis=1)


# In[204]:


def number_12board(row):
  text = row['board12']
  return dic_12board[text]
df['12board'] = df.apply(number_12board,axis=1)


# In[205]:


def number_degree(row):
  text = row['Degree']
  return dic_degree[text]
df['Degree'] = df.apply(number_degree,axis=1)


# In[206]:


def number_collegestate(row):
  text = row['CollegeState']
  return dic_collegestate[text]
df['CollegeState'] = df.apply(number_collegestate,axis=1)


# In[207]:


sampledf=df[['ID', 'Gender', 'DOB', '10percentage', '10board', '12graduation',
       '12percentage', '12board', 'CollegeID', 'CollegeTier', 'Degree',
       'Specialization', 'collegeGPA', 'CollegeCityID', 'CollegeCityTier',
       'CollegeState', 'GraduationYear', 'English', 'Logical', 'Quant',
       'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon',
       'ComputerScience', 'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg',
       'CivilEngg', 'conscientiousness', 'agreeableness', 'extraversion',
       'nueroticism', 'openess_to_experience', 'HighSalary']]


# In[208]:


sampledf.dtypes


# In[209]:


df.apply(lambda x: sum(x.isna()/len(df)))


# In[210]:


logistic_regression = LogisticRegression(max_iter=10000)


# In[211]:


import numpy as np
x_train, x_test, y_train, y_test = train_test_split(sampledf[[ '10percentage','10board',
       '12percentage', '12board', 'CollegeTier', 'Degree','GraduationYear',
       'Specialization', 'collegeGPA', 'CollegeCityTier','English', 'Logical', 'Quant',
       'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon',
       'ComputerScience', 'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg',
       'CivilEngg','conscientiousness', 'agreeableness', 'extraversion',
       'nueroticism', 'openess_to_experience']].to_numpy(),sampledf['HighSalary'], test_size=0.2,random_state=42)


# In[212]:


from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
x_train = sc_x.fit_transform(x_train)  
x_test = sc_x.transform(x_test)


# In[213]:


logistic_regression.fit(x_train,y_train)


# In[214]:


y_pred = logistic_regression.predict(x_test)


# In[218]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[219]:


accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy


# In[220]:


accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
accuracy_percentage


# In[217]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




