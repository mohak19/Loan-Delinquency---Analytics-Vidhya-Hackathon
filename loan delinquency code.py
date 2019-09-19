#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import train file:

data = pd.read_csv("C:/ml projects/av ml hackathon/train.csv")


#import test file:


testfiledata = pd.read_csv("C:/ml projects/av ml hackathon/test.csv")


data['number_of_borrowers'] = np.where((data['number_of_borrowers']) > 1,1,0)
testfiledata['number_of_borrowers'] = np.where((testfiledata['number_of_borrowers']) > 1,1,0)


# In[4]:


def recc(row):
    if row['financial_institution'] == 'OTHER':
        return 1
    elif row['financial_institution'] == 'Browning-Hart':
        return 2
    else:
        return 3

data['financial_institution'] = data.apply(recc, axis=1)
testfiledata['financial_institution'] = testfiledata.apply(recc, axis=1)


# In[5]:


#label encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['source'] = le.fit_transform(data['source'])
data['loan_purpose'] = le.fit_transform(data['loan_purpose'])
data['insurance_type'] = le.fit_transform(data['insurance_type']) #gets converted into binary variable
testfiledata['source'] = le.fit_transform(testfiledata['source'])
testfiledata['loan_purpose'] = le.fit_transform(testfiledata['loan_purpose'])
testfiledata['insurance_type'] = le.fit_transform(testfiledata['insurance_type'])


# In[6]:


from pandas import datetime
data['origination_date'] = pd.to_datetime(data['origination_date'],infer_datetime_format = True)
data['first_payment_date'] = pd.to_datetime(data['first_payment_date'],infer_datetime_format = True)
from datetime import datetime
testfiledata['origination_date'] = pd.to_datetime(testfiledata['origination_date'],infer_datetime_format = True)
testfiledata['first_payment_date'] = pd.to_datetime(testfiledata['first_payment_date'],format = '%b-%y')


# In[7]:


data['orig_date_month'] = data['origination_date'].dt.month
data['first_pmt_date_month'] = data['first_payment_date'].dt.month

testfiledata['orig_date_month'] = testfiledata['origination_date'].dt.month
testfiledata['first_pmt_date_month'] = testfiledata['first_payment_date'].dt.month


# In[8]:


#onehotencoding data
cat_cols = ['source',
'financial_institution','loan_purpose']

data1 = data[cat_cols]

from sklearn.preprocessing import OneHotEncoder

X = data1.values

sources = np.unique(X[:,0])
finstitutions = np.unique(X[:,1])
lpurposes = np.unique(X[:,2])

ohe = OneHotEncoder(categories=[sources,finstitutions,lpurposes])
X = ohe.fit_transform(X).toarray()

data1 = pd.DataFrame(data = X)
data = pd.concat([data, data1], axis=1)


# In[9]:


#onehotencoding testfiledata
data2 = testfiledata[cat_cols]

X = data2.values

sources = np.unique(X[:,0])
finstitutions = np.unique(X[:,1])
lpurposes = np.unique(X[:,2])

ohe = OneHotEncoder(categories=[sources,finstitutions,lpurposes])
X = ohe.fit_transform(X).toarray()

data2 = pd.DataFrame(data = X)
testfiledata = pd.concat([testfiledata, data2], axis=1)


# In[10]:


data.columns = data.columns.map(str)
testfiledata.columns = testfiledata.columns.map(str)


# In[11]:


#first feature creation : first_payment_Date - origination date
data['first_pay-org_date'] = pd.Series(data['first_payment_date'] - data['origination_date'])
testfiledata['first_pay-org_date'] = pd.Series(testfiledata['first_payment_date'] - testfiledata['origination_date'])

data['first_pay-org_date'] = data['first_pay-org_date'].dt.days
testfiledata['first_pay-org_date'] = testfiledata['first_pay-org_date'].dt.days


# In[12]:


#second feature creation: no of defaults in m1 to m12
data['no_of_defaults'] = data['m1'] + data['m2'] + data['m3'] + data['m4'] + data['m5'] + data['m6'] + data['m7'] +data['m8'] +data['m9'] +data['m10'] +data['m11'] +data['m12']
testfiledata['no_of_defaults'] = testfiledata['m1'] + testfiledata['m2'] + testfiledata['m3'] + testfiledata['m4'] + testfiledata['m5'] + testfiledata['m6'] + testfiledata['m7'] +testfiledata['m8'] +testfiledata['m9'] +testfiledata['m10'] +testfiledata['m11'] +testfiledata['m12']


# In[13]:


#third feature creation : any default - binary variable
data['any_default'] = np.where((data['m1'] + data['m2'] + data['m3'] + data['m4'] + data['m5'] + data['m6'] + data['m7'] +data['m8'] +data['m9'] +data['m10'] +data['m11'] +data['m12']) > 0, 1,0)
testfiledata['any_default'] = np.where((testfiledata['m1'] + testfiledata['m2'] + testfiledata['m3'] + testfiledata['m4'] + testfiledata['m5'] + testfiledata['m6'] + testfiledata['m7'] +testfiledata['m8'] +testfiledata['m9'] +testfiledata['m10'] +testfiledata['m11'] +testfiledata['m12']) > 0,1,0)


# In[14]:


#4th,5th,6th,7th feature creation: slope from m1 to m3, m4 to m6 and so on
data['slope1m3'] = data['m3'] - data['m1']
data['slope4m6'] = data['m6'] - data['m4']
data['slope7m9'] = data['m9'] - data['m7']
data['slope10m12'] = data['m12'] - data['m10']

testfiledata['slope1m3'] = testfiledata['m3'] - testfiledata['m1']
testfiledata['slope4m6'] = testfiledata['m6'] - testfiledata['m4']
testfiledata['slope7m9'] = testfiledata['m9'] - testfiledata['m7']
testfiledata['slope10m12'] = testfiledata['m12'] - testfiledata['m10']


# In[15]:


#eighth feature creation: perhaps the most important of all:
#to_be_paid = unpaid(1+rate)^loan_term_left
#creating variable loan_term_left

data['loan_term_left'] = ((data['loan_term']/30) - data['orig_date_month'])
testfiledata['loan_term_left'] = ((testfiledata['loan_term']/30) - testfiledata['orig_date_month'])

data['to_be_paid'] = (data['unpaid_principal_bal'])*((data['interest_rate'] + 1)**data['loan_term_left'])
testfiledata['to_be_paid'] = (testfiledata['unpaid_principal_bal'])*((testfiledata['interest_rate'] + 1)**testfiledata['loan_term_left'])


# In[16]:


#now i'll try making a new variable: no of months of defaults

def oner(val):
    return np.where(val > 0,1,0)

data['no_month_def'] = oner(data['m1']) + oner(data['m2']) + oner(data['m3']) + oner(data['m4']) + oner(data['m5']) + oner(data['m6']) + oner(data['m7']) + oner(data['m8']) + oner(data['m9']) + oner(data['m10']) + oner(data['m11']) + oner(data['m12'])
testfiledata['no_month_def'] = oner(testfiledata['m1']) + oner(testfiledata['m2']) + oner(testfiledata['m3']) + oner(testfiledata['m4']) + oner(testfiledata['m5']) + oner(testfiledata['m6']) + oner(testfiledata['m7']) + oner(testfiledata['m8']) + oner(testfiledata['m9']) + oner(testfiledata['m10']) + oner(testfiledata['m11']) + oner(testfiledata['m12'])

#another variable: no of months default 3-month-pairwise

data['1any3'] = np.where((oner(data['m1']) + oner(data['m2']) + oner(data['m3'])) > 0,1,0)
data['4any6'] = np.where((oner(data['m4']) + oner(data['m5']) + oner(data['m6'])) > 0,1,0)
data['7any9'] = np.where((oner(data['m7']) + oner(data['m8']) + oner(data['m9'])) > 0,1,0)
data['10any12'] = np.where((oner(data['m10']) + oner(data['m11']) + oner(data['m12'])) > 0,1,0)
testfiledata['1any3'] = np.where((oner(testfiledata['m1']) + oner(testfiledata['m2']) + oner(testfiledata['m3'])) > 0,1,0)
testfiledata['4any6'] = np.where((oner(testfiledata['m4']) + oner(testfiledata['m5']) + oner(testfiledata['m6'])) > 0,1,0)
testfiledata['7any9'] = np.where((oner(testfiledata['m7']) + oner(testfiledata['m8']) + oner(testfiledata['m9'])) > 0,1,0)
testfiledata['10any12'] = np.where((oner(testfiledata['m10']) + oner(testfiledata['m11']) + oner(testfiledata['m12'])) > 0,1,0)

#another variable : months since last default

def rec(row):
    if row['m12'] > 0:
        return 0
    elif row['m11'] > 0:
        return 1
    elif row['m10'] > 0:
        return 2
    elif row['m9'] > 0:
        return 3
    elif row['m8'] > 0:
        return 4
    elif row['m7'] > 0:
        return 5
    elif row['m6'] > 0:
        return 6
    elif row['m5'] > 0:
        return 7
    elif row['m4'] > 0:
        return 8
    elif row['m3'] > 0:
        return 9
    elif row['m2'] > 0:
        return 10
    elif row['m1'] > 0:
        return 11
    else:
        return 24

data['months_since_def'] = data.apply(rec, axis=1)
testfiledata['months_since_def'] = testfiledata.apply(rec, axis=1)


# In[17]:


#creating another variable : monthly installment
#installment =  [P x R x (1+R)^N]/[(1+R)^N-1]

data['installment'] = ((data['to_be_paid'])/((1 + data['interest_rate'])**(data['loan_term_left'] - 1)))
testfiledata['installment'] = ((testfiledata['to_be_paid'])/((1 + testfiledata['interest_rate'])**(testfiledata['loan_term_left'] - 1)))


# In[18]:


#continuous variable normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cont_cols = ['interest_rate',
'unpaid_principal_bal',
'loan_term',
'loan_to_value',
'number_of_borrowers',
'debt_to_income_ratio',
'borrower_credit_score',
'insurance_percent',
'co-borrower_credit_score',
'to_be_paid',
'no_of_defaults',
'first_pay-org_date',
'loan_term_left',
'1any3','4any6','7any9','10any12',
'slope1m3','slope4m6','slope7m9','slope10m12',
'orig_date_month','first_pmt_date_month',
'no_month_def',
'months_since_def',
'installment']
data[cont_cols] = scaler.fit_transform(data[cont_cols])
testfiledata[cont_cols] = scaler.fit_transform(testfiledata[cont_cols])


# In[20]:


#now using stratified kfold validation with random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
all_features = [
#'loan_id', 
#'source', 
#'financial_institution', 
#'interest_rate',
#'unpaid_principal_bal', 
#'loan_term', 
#'origination_date',
#'first_payment_date', 
'loan_to_value', 
'number_of_borrowers',
'debt_to_income_ratio', 
'borrower_credit_score', 
'loan_purpose',
'insurance_percent', 
'co-borrower_credit_score', 
'insurance_type', 
'm1',
'm2', 
'm3', 
'm4', 
'm5', 
'm6', 
'm7', 
'm8', 
'm9', 
'm10', 
'm11', 
'm12',
#'m13', 
#'orig_date_month', 
#'first_pmt_date_month', 
'0', 
'1', 
'2', 
'3',
'4', 
'5', 
'6', 
'7', 
'8', 
'first_pay-org_date', 
'no_of_defaults',
'any_default', 
'slope1m3', 
'slope4m6', 
'slope7m9', 
'slope10m12',
'loan_term_left', 
'to_be_paid', 
'no_month_def', 
'1any3', 
'4any6',
'7any9', 
'10any12', 
'months_since_def', 
'installment'
]

X = data[all_features]

y = data['m13']

xtest = testfiledata[all_features]

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

kf = StratifiedKFold(n_splits=9,shuffle=True,random_state=67)
pred_test_full =0
cv_score =[]
i=1
for train_index,test_index in kf.split(X,y):
    print('{} of KFold {}'.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y.loc[train_index],y.loc[test_index]
    
    #model
    lr = RandomForestClassifier(n_estimators = 105, max_depth = 4, min_samples_split = 44,min_samples_leaf = 22,max_features = 'auto',)
    lr.fit(xtr,ytr)
    score = f1_score(yvl,lr.predict(xvl))
    print("this fold score: ", score)
    pred_test = lr.predict_proba(xvl)[:,1]
    i += 1


proba = lr.predict_proba(xtest)[:,1]
proba = pd.Series(proba)
proba = np.where((proba > 0.33),1,0)

col = testfiledata['loan_id']
rf1 = pd.DataFrame()
rf1['loan_id'] = col
rf1['m13'] = proba

rf1.to_csv("C:/ml projects/av ml hackathon/rf_python.csv",index=False)




