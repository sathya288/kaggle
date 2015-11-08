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
from sklearn.linear_model import RidgeCV
import zipfile
import time 

gStoreTypes=['StoreType=a','StoreType=b','StoreType=c','StoreType=d']
gPromoInterval=['PromoInterval=0', 'PromoInterval=Feb,May,Aug,Nov','PromoInterval=Jan,Apr,Jul,Oct', 'PromoInterval=Mar,Jun,Sept,Dec']
gAssortment=['Assortment=a', 'Assortment=b','Assortment=c']
gSingle=[]

def loadData(trainfile,testfile,storefile):
  print('loading data ...')
  ftrain=zipfile.ZipFile(trainfile)
  dtrain=pd.read_csv(ftrain.open('train.csv'),parse_dates=['Date'],low_memory=False)
  
  fstore=zipfile.ZipFile(storefile)
  dstore=pd.read_csv(fstore.open('store.csv'))

  ftest=zipfile.ZipFile(testfile)
  dtest=pd.read_csv(ftest.open('test.csv'),parse_dates=['Date'])

  print('loading data ...completed')
  return dtrain, dtest,dstore

def sanitizeData(train, test,store):

  print('sanitizing data ...')
  xtrain=sanitizeTrain(train)
  xstore=sanitizeStore(store)
  xtest=sanitizeTest(test)

  #handle NaNs, do transformations and prepare the data for further processing.
  dtrain = pd.merge(xtrain,xstore,on='Store')
  dtest= pd.merge(xtest,xstore,on='Store')
  print('sanitizing data ... completed')
  return dtrain, dtest

def sanitizeTrain(train):
  print('sanitizing Training data ... ')
  train['Day']=train['Date'].apply(lambda x:x.day)
  train['Month']=train['Date'].apply(lambda x:x.month)
  train['Year']=train['Date'].apply(lambda x:x.year)
  train['LogSales']=train['Sales'].apply(lambda x:math.log(x+1))
  train['StateHoliday'].replace({'0':0,'a':1,'b':2,'c':3},inplace=True)
  print('sanitizing Training data ... completed')
  return train

def sanitizeTest(test):
  print('sanitizing Test data ... ')
  test['Open'].replace('NaN',1,inplace=True)
  test['StateHoliday'].replace({'0':0,'a':1,'b':2,'c':3},inplace=True)
  test['Day']=train['Date'].apply(lambda x:x.day)
  test['Month']=train['Date'].apply(lambda x:x.month)
  test['Year']=train['Date'].apply(lambda x:x.year)
  print('sanitizing Test data ... completed')
  return test

def sanitizeStore(store):
  print('sanitizing Store data ... ')

  store['CompetitionDistance'].fillna(1,inplace=True)
  store['CompetitionOpenSinceMonth'].fillna(0,inplace=True)
  store['CompetitionOpenSinceYear'].fillna(0,inplace=True)
  store['Promo2SinceWeek'].fillna(0,inplace=True)
  store['Promo2SinceYear'].fillna(0,inplace=True)
  store['PromoInterval'].fillna('0',inplace=True)
  Imputer(missing_values='NaN',strategy='most_frequent',copy=False).fit_transform(store['CompetitionOpenSinceMonth'])
  Imputer(missing_values='NaN',strategy='most_frequent',copy=False).fit_transform(store['CompetitionOpenSinceYear'])
  store['LogCompDist']=store['CompetitionDistance'].apply(lambda x:math.log(x))
  store['LogCompDays']=store.apply(func=getDays,axis=1)
  print('sanitizing Store data ... completed')
  return encode_onehot(store,['StoreType','Assortment','PromoInterval'])

def getDays(row):
  month=row['CompetitionOpenSinceMonth']
  year=row['CompetitionOpenSinceYear']
  if (month==0 or year==0):
    return 1
  #approx days will do.
  return math.log((12-month)*30 + (2016-year)*12*30)

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
  print('starting feature engg ...')
  #columns to be dropped based on simple correlation
  traindropcols=['Sales','CompetitionDistance','Customers','Date','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']
  testdropcols=['Date','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']
  train.drop(traindropcols,axis=1,inplace=True)
  test.drop(testdropcols,axis=1,inplace=True)
  print('starting feature engg ... completed')
  return train, test

def GBModel(train,test,splitcriteria):
  train.reindex(np.random.permutation(train.index))
  trA,trB,trC,trD=splitModels(train,splitcriteria)
  
  print('starting Gradient Boosting ...')
  trA_X=trA.drop(['LogSales'],axis=1)
  trA_Y=trA['LogSales']
  modelA=GradientBoostingRegressor(n_estimators=200,max_depth=9,min_samples_leaf=7,min_samples_split=7,warm_start=True)
  modelA.fit(trA_X,trA_Y)
  
  trB_X=trB.drop(['LogSales'],axis=1)
  trB_Y=trB['LogSales']
  modelB=GradientBoostingRegressor(n_estimators=200,max_depth=9,min_samples_leaf=7,min_samples_split=7,warm_start=True)
  modelB.fit(trB_X,trB_Y)
  
  trC_X=trC.drop(['LogSales'],axis=1)
  trC_Y=trC['LogSales']
  modelC=GradientBoostingRegressor(n_estimators=200,max_depth=9,min_samples_leaf=7,min_samples_split=7,warm_start=True)
  modelC.fit(trC_X,trC_Y)
  
  trD_X=trD.drop(['LogSales'],axis=1)
  trD_Y=trD['LogSales']
  modelD=GradientBoostingRegressor(n_estimators=200,max_depth=9,min_samples_leaf=7,min_samples_split=7,warm_start=True)
  modelD.fit(trD_X,trD_Y)

  print('completed Gradient Boosting ...')
  print('predicting ...')
  teA,teB,teC,teD=splitModels(test,gPromoInterval)
  teA_y=getDF(modelA.predict(teA.drop(['Id'],axis=1)))
  teB_y=getDF(modelB.predict(teB.drop(['Id'],axis=1)))
  teC_y=getDF(modelC.predict(teC.drop(['Id'],axis=1)))
  teD_y=getDF(modelD.predict(teD.drop(['Id'],axis=1)))
  pd.concat([teA,teB,teC,teD])['Id'].to_csv('Inputs.csv')
  pd.concat([teA_y,teB_y,teC_y,teD_y]).to_csv('Predicts.csv')
  print('predicting ... done')

def GBModel2(train,test,splitcriteria):
  train.reindex(np.random.permutation(train.index))
  trains=splitModels(train,splitcriteria)
  print('starting Gradient Boosting ...')
  print(splitcriteria)
  models=[]
  for train in trains:
    trA_X=train.drop(['LogSales'],axis=1)
    trA_Y=train['LogSales']
    model=GradientBoostingRegressor(n_estimators=500,max_depth=9,min_samples_leaf=7,min_samples_split=7,warm_start=True)
    model.fit(trA_X,trA_Y)
    models.append(model)
  
  print('completed Gradient Boosting ...')
  print('predicting ...')
  tests=splitModels(test,gPromoInterval)
  preds=[]
  for model, test in zip(models,tests):
    preds.append(getDF(model.predict(test.drop(['Id'],axis=1))))
  pd.concat(tests)['Id'].to_csv('Inputs'+str(time.time())+'.csv')
  pd.concat(preds).to_csv('Predicts.csv'+str(time.time())+'.csv')
  print('predicting ... done')

def getDF(testY):
  t=[]
  for y in testY:
    t.append(math.exp(y))
  return pd.DataFrame(t)

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


def RidgeCVLinear(train,test):
  print('starting RidgeCVLinear ...')
  ridge=RidgeCV(normalize=True,cv=5)
  train.reindex(np.random.permutation(train.index))
  tr_X=train.drop(['LogSales'],axis=1)
  tr_Y=train['LogSales']
  cutoff=math.floor(0.7*tr_Y.size)
  ridge.fit(tr_X[:cutoff],tr_Y[:cutoff])
  predY=ridge.predict(tr_X[cutoff:])
  mspe=rmspe(predY,tr_Y[cutoff:])
  print('rmspe is %9f'% mspe)
  print(train.columns)
  print(ridge.coef_)
  print('starting RidgeCVLinear ... completed')
  return ridge

def splitModels(train,cond):
  #split based on storeType
  print('splitting models ...')
  if len(cond)==0:
    return [train]
  return [train[train[x]==1] for x in cond]
  print('splitting models ... completed')

train,test,store=loadData('data/train.csv.zip', 'data/test.csv.zip','data/store.csv.zip')
dtrain,dtest=sanitizeData(train,test,store)
dtrain,dtest=feature_engg(dtrain,dtest)
#GBModel(dtrain,dtest,gStoreType)
GBModel2(dtrain,dtest,gStoreType)
#ridge=RidgeCVLinear(dtrain,dtest)
