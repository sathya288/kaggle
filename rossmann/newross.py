'''

This is a new version to attempt the Kaggle Rossmann prediction competition.

__author__ = Sathya, AR
__version__ = 3.14
__date__ = 7th Nov 2015

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
import zipfile

def loadData(trainfile,testfile,storefile):
  ftrain=zipfile.ZipFile(trainfile)
  dtrain=pd.read_csv(ftrain.open('train.csv'),parse_dates=['Date'],low_memory=False)
  
  fstore=zipfile.ZipFile(storefile)
  dstore=pd.read_csv(fstore.open('store.csv'))

  ftest=zipfile.ZipFile(testfile)
  dtest=pd.read_csv(ftest.open('test.csv'),parse_dates=['Date'])

  return dtrain, dtest,dstore

def sanitizeData(train, test,store):

  xtrain=sanitizeTrain(train)
  xstore=sanitizeStore(store)
  xtest=sanitizeTest(test)

  #handle NaNs, do transformations and prepare the data for further processing.
  dtrain = pd.merge(xtrain,xstore,on='Store')
  dtest= pd.merge(xtest,xstore,on='Store')

  return dtrain, dtest

def sanitizeTrain(train):
  train['Day']=train['Date'].apply(lambda x:x.day)
  train['Month']=train['Date'].apply(lambda x:x.month)
  train['Year']=train['Date'].apply(lambda x:x.year)
  train['LogSales']=train['Sales'].apply(lambda x:math.log(x+1))
  train['StateHoliday'].replace({'0':0,'a':1,'b':2,'c':3},inplace=True)
  return train

def sanitizeTest(test):
  test['Open'].replace('NaN',1,inplace=True)
  test['StateHoliday'].replace({'0':0,'a':1,'b':2,'c':3},inplace=True)
  return test

def sanitizeStore(store):

  store['CompetitionDistance'].fillna(1,inplace=True)
  store['Promo2SinceWeek'].fillna(0,inplace=True)
  store['Promo2SinceYear'].fillna(0,inplace=True)
  store['PromoInterval'].fillna('0',inplace=True)
  Imputer(missing_values='NaN',strategy='most_frequent',copy=False).fit_transform(store['CompetitionOpenSinceMonth'])
  Imputer(missing_values='NaN',strategy='most_frequent',copy=False).fit_transform(store['CompetitionOpenSinceYear'])
 
  return encode_onehot(store,['StoreType','Assortment','PromoInterval'])

def encode_onehot(df, cols):
  """
    One-hot encoding is applied to columns specified in a pandas DataFrame.

    Modified from: https://gist.github.com/kljensen/5452382

    Details:

    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
  """
  vec = DictVectorizer()
  vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
  vec_data.columns = vec.get_feature_names()
  vec_data.index = df.index

  df = df.drop(cols, axis=1)
  df = df.join(vec_data)
  return df

def feature_engg(train, test):

  #columns to be dropped based on simple correlation
  traindropcols=['Sales','Customers','Store','Day','Month','Year','Date','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']
  testdropcols=['Store','Date','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']
  train.drop(traindropcols,axis=1,inplace=True)
  test.drop(testdropcols,axis=1,inplace=True)
  return train, test

def GBModel(train,test):
  train.reindex(np.random.permutation(train.index))
  tr_X=train.drop(['LogSales'],axis=1)
  tr_Y=train['LogSales']
  cutoff=math.floor(0.7*tr_Y.size)
  model=GradientBoostingRegressor(n_estimators=300,max_depth=9,min_samples_leaf=7,min_samples_split=7,warm_start=True)
  model.fit(tr_X[:cutoff],tr_Y[:cutoff])
  predY=model.predict(tr_X[cutoff:])
  mspe=rmspe(predY,tr_Y[cutoff:])
  print('rmspe is %9f'% mspe)
  model2=GradientBoostingRegressor(n_estimators=300,max_depth=9,min_samples_leaf=7,min_samples_split=7,warm_start=True)
  model2.fit(tr_X, tr_Y)
  testY=model2.predict(test.drop(['Id'],axis=1))
  t=[]
  for y in testY:
    t.append(math.exp(y))
  preds=pd.DataFrame(t)
  test['Id'].to_csv('Inputs.csv')
  preds.to_csv('Predicts.csv')

# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
  w = np.zeros(y.shape, dtype=float)
  ind = y != 0
  w[ind] = 1./(y[ind]**2)
  return w

def rmspe(yhat, y):
  w = ToWeight(y)
  mspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
  return mspe

train,test,store=loadData('data/train.csv.zip', 'data/test.csv.zip','data/store.csv.zip')
dtrain,dtest=sanitizeData(train,test,store)
dtrain,dtest=feature_engg(dtrain,dtest)
GBModel(dtrain,dtest)
