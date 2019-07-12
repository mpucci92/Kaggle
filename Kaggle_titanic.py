#!/usr/bin/env python
# coding: utf-8

# In[4]:


### Import all libraries ### 

import pandas as pd
import numpy as np
import random as rnd

# Visualizations # 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Machine learning #
from sklearn.linear_model import LogisticRegression;
from sklearn.svm import SVC, LinearSVC;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
from sklearn.linear_model import Perceptron;
from sklearn.linear_model import SGDClassifier;
from sklearn.tree import DecisionTreeClassifier;
from pandas.plotting import scatter_matrix;
from sklearn.grid_search import GridSearchCV;
from sklearn.metrics import accuracy_score;
from sklearn.model_selection import train_test_split;
from sklearn.decomposition import PCA;
from sklearn.metrics import confusion_matrix;
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis;
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis;
import warnings;
warnings.filterwarnings("ignore");
warnings.filterwarnings("ignore", category=DeprecationWarning); 
from sklearn.model_selection import cross_val_score;
from sklearn.ensemble import GradientBoostingClassifier;


# In[5]:


### Acquiring the data ###

train = pd.read_csv("C:\\Users\\mpucci\\Desktop\\train.csv")
test = pd.read_csv("C:\\Users\\mpucci\\Desktop\\test.csv")


# In[31]:


### Understanding your data ###

train.describe() #-> Shows that not all of the columns have the same amount of values, Age does not have 891 values. 

#Comments: Survived is a categorical feature, 0 and 1. Training set samples/ pop. is 40%. Pretty representative sample. 


# In[7]:


train.describe(include=['O']) # Run a describe function on Categorical Data 


# In[8]:


train.info()
#test.info()


# In[9]:


# train.columns.values -> displays column titles 


# In[10]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[11]:


train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[12]:


train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[13]:


train[['Parch','Survived']].groupby(["Parch"],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[14]:


#train.groupby(['Age']).nunique() #Unique values in the Age,Fare column 
#train.groupby(['Fare']).nunique()


# In[15]:


scatter_matrix(train,alpha=0.2, figsize=(6, 6), diagonal='kde'); #Perform a log transform on the axis for Fare and Parch, Expand the distribution


# In[16]:


# train_df = pd.DataFrame(train) dataframe version 


# In[17]:


age_histogram = sns.FacetGrid(train,col="Survived")
age_histogram.map(plt.hist,'Age',bins=20)


# In[18]:


grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20);


# In[19]:


train['Sex'] = train['Sex'].factorize()[0] #Male = 0, Female = 1. 


# In[20]:


train.columns.values


# In[21]:


test.columns.values
test['Sex'] = test['Sex'].factorize()[0] #Male = 0, Female = 1. 


# In[22]:


train.isnull().sum() # total NaN per column
# train.isnull().sum().sum() -> Total amount of NaN values in data set


# In[23]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[26]:


mode = train['Embarked'].dropna().mode()[0]
train['Embarked'] = train['Embarked'].fillna(mode,inplace=True)
train['Embarked'] = train['Embarked'].map({'S':0,'C':1,'Q':2})


# In[27]:


median_values = train["Age"].median()
print (median_values)
train["Age"] = train["Age"].fillna(median_values, inplace=True)


# In[30]:


train.isnull().sum()


# In[67]:


train['Age'].describe()
train[["Age"]].boxplot()


# In[159]:


train.head()
train.Ticket.unique()


# In[120]:


test.head()


# In[121]:


train.shape


# In[122]:


test.shape


# In[123]:


df_train = train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
df_test = test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


# In[144]:


median_values = test["Age"].median()
print (median_values)
df_test["Age"] = test["Age"].fillna(median_values)
df_train['Embarked'] = df_train['Embarked'].fillna(1)
df_train.isnull().sum()


# In[ ]:


#Odd and even numbered tickets - Create a feature - maybe has some predictive power. 
#Prefix on the ticket or not? - > could develop a feature to see if has some predictive power 


# In[145]:


df_train


# In[147]:


df_test['Embarked'] = df_test['Embarked'].map({'S':0,'C':1,'Q':2})
df_test.tail()


# In[158]:



#df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)







df_trainX = df_train.iloc[:,1:]
df_trainY = df_train.iloc[:,0]

#print(df_trainX.isnull().sum())
X_train, X_test, y_train, y_test = train_test_split(df_trainX, df_trainY, test_size = 0.2, random_state=1,shuffle=True)
#print(X_train)


import seaborn as sns
corr = df_trainX.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,annot=True)


# In[149]:


#Model 1 : PCA with LDA 
pca = PCA(n_components=4)
pca.fit(X_train)

Z0 = pca.fit_transform(X_train)
Z1 = pca.transform(X_test)


# In[150]:


pca_ldascores = []
lda = LinearDiscriminantAnalysis()

#y= ziptrain[:,0]         # test real y values
#y_values = ziptest[:,0]  # validation real y values

lda.fit(Z0,y_train)                   #LDA FIT
predictions = lda.predict(Z1)   #LDA Predictions
y_hat = predictions

score = (accuracy_score(y_test, y_hat))
pca_ldascores.append(score)
print(score)


# In[151]:


pca_qdascores = []
qda = QuadraticDiscriminantAnalysis()

#y= ziptrain[:,0]         # test real y values
#y_values = ziptest[:,0]  # validation real y values

qda.fit(Z0,y_train)                   #LDA FIT
predictions = qda.predict(Z1)   #LDA Predictions
y_hat = predictions

score = (accuracy_score(y_test, y_hat))
pca_qdascores.append(score)
print(score)


# In[152]:


## Model 3 - Random Forest Classifier ## 

rfc = RandomForestClassifier()

parameters = {
    "max_depth": [i for i in range(1,7,1)],
    "criterion":['entropy'],
    "max_features":["sqrt"],
    "n_estimators":[i for i in range(1,6,1)],
    #'verbose':[(2)]                                                        # Used for logging - visual construction of tree
    }

rfc_model = GridSearchCV(rfc, parameters, cv=10)


# In[153]:


rfc_model.fit(X_train, y_train)
print(rfc_model.best_params_) 


# In[154]:


rfc = RandomForestClassifier(max_depth=5,max_features="sqrt",criterion="entropy",n_estimators=5)
rfc.fit(X_train, y_train)
y_hat = rfc.predict(X_test)
print(accuracy_score(y_test, y_hat))


# In[155]:


## Model 4 - GradientBoosting Classifier ##

gradboost = GradientBoostingClassifier(n_estimators=5,max_depth=5,max_features="sqrt", random_state=1)
gradboost.fit(X_train, y_train)
y_hat = gradboost.predict(X_test)
insample_score = accuracy_score(y_test, y_hat)

print("Insample accuracy score for Gradient Boosting Classifier")
print(insample_score)


# In[ ]:





# In[ ]:




