#!/usr/bin/env python
# coding: utf-8

# # Diabetes prediction with SVM¶

# In[41]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,StandardScaler # for standardization
from sklearn.model_selection import train_test_split, GridSearchCV ,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score, mean_squared_error, r2_score,classification_report,roc_auc_score,roc_curve
from sklearn.svm import SVC


# In[42]:


# turn off alerts
from warnings import filterwarnings
filterwarnings ('ignore')


# amaç: Hastanemizde tutulan veri setinde kişilerin bazı bilgileri bulunmaktadır.Kişinin tahlil sonuçlarına göre şeker hastası olup olmadığına dair bir tahminleme modeli gerçekleştirmemiz isteniyor.

# In[43]:


df= pd.read_csv("diabetes.csv")


# In[44]:


df.head()


# # Model ve Tahmin

# In[45]:


df["Outcome"].value_counts() # representation numbers of the dependent variable.


# There is information of 268 people in the data 1, that is, the number of diabetics, and information of 500 people from the data of 0, that is, the data of people who do not have diabetes.

# In[46]:


df.describe().T # descriptive statistics


# In[47]:


y=df["Outcome"]# get dependent variable
X=df.drop(["Outcome"], axis=1) # bağımsız değişkenleri alınması
X_train,X_test,y_train,y_test = train_test_split(X,# independent variable
                                                y, #the dependent variable
                                                test_size=0.30,# test verisi
                                                random_state=42) 


# In[48]:


svm_model=SVC(kernel="linear").fit(X_train,y_train)# model installed


# In[49]:


y_pred = svm_model.predict(X) # predictive acquisition values


# In[50]:


accuracy_score(y,y_pred) # success rate


# In[51]:


print(classification_report(y,y_pred)) #detailed reporting


# # Model Tuning

# In[52]:


svm_params={"C": np.arange(1,10),
            "kernel": ["linear","rbf"]} #grouping of parameters


# In[53]:


svm_model=SVC()# model object


# In[55]:


# finding ideal parameter values
svm_cv_model=GridSearchCV(svm_model,svm_params,cv=5,n_jobs=-1,verbose=2).fit(X_train,y_train).fit(X_train,y_train)


# In[56]:


#best model success values


# In[57]:


#the most ideal parameters
svm_cv_model.best_params_


# In[61]:



svm_tuned= SVC(C=2,kernel="linear").fit(X_train, y_train)


# In[59]:


y_pred=svm_tuned.predict(X_test)


# In[60]:


accuracy_score(y_test,y_pred)


# In[ ]:




