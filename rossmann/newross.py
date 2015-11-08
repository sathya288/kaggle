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
from time import strftime

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
  xtrain=sanitizeInputs(train)
  xstore=sanitizeStore(store)
  xtest=sanitizeInputs(test)

  #handle NaNs, do transformations and prepare the data for further processing.
  dtrain = pd.merge(xtrain,xstore,how='inner',on='Store')
  dtest= pd.merge(xtest,xstore,how='inner',on='Store')
  #catcols=['DayOfWeek','Promo','Store','Month','Day','Year','StoreType']
  catcols=['DayOfWeek','Promo','Store','Month','Day','Year']
  dtrain=encode_onehot(dtrain,catcols)
  dtest=encode_onehot(dtest,catcols)
  print('sanitizing data ... completed')
  return dtrain, dtest

def sanitizeInputs(train):
  print('sanitizing Training data ... ')
  train['Open'].replace('NaN',1,inplace=True)
  train['Day']=train['Date'].apply(lambda x:x.day)
  train['Month']=train['Date'].apply(lambda x:x.month)
  train['Year']=train['Date'].apply(lambda x:x.year)
  if 'Sales' in train.columns:
    train=train[train['Sales']>0]
    train['LogSales']=train['Sales'].apply(lambda x:math.log(x+1))
  train['StateHoliday'].replace({'0':0,'a':1,'b':2,'c':3},inplace=True)
  print('sanitizing Training data ... completed')
  return train

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
  #return encode_onehot(store,['StoreType'])
  #return store

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
  print('feature engg ...')
  #columns to be dropped based on simple correlation
  #catcols=['DayOfWeek','Promo','Store','Month','Day','Year','StoreType']
  #dropcols=['Promo2', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval','Date','Open','StateHoliday','SchoolHoliday','Assortment','CompetitionDistance']
  #The above two lines were from previous run which fetched the highest score till date. So copying here for reference.
  #train['IsPromo2On']=train.apply(func=getPromo2,axis=1)
  #test['IsPromo2On']=test.apply(func=getPromo2,axis=1)
  dropcols=['Day','Month','Year','Open','Date','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear']
  traindropcols=['Sales','Customers']
  train.drop(traindropcols,axis=1,inplace=True)
  train.drop(dropcols,axis=1,inplace=True)
  test.drop(dropcols,axis=1,inplace=True)
  print('feature engg ... completed')
  return train, test

def getPromo2(row):
  m=row['Date'].month
  months={0:'PromoInterval=0', 1:'PromoInterval=Jan,Apr,Jul,Oct',4:'PromoInterval=Jan,Apr,Jul,Oct',7:'PromoInterval=Jan,Apr,Jul,Oct',10:'PromoInterval=Jan,Apr,Jul,Oct',2:'PromoInterval=Feb,May,Aug,Nov',5:'PromoInterval=Feb,May,Aug,Nov',8:'PromoInterval=Feb,May,Aug,Nov',11:'PromoInterval=Feb,May,Aug,Nov',3:'PromoInterval=Mar,Jun,Sept,Dec',6:'PromoInterval=Mar,Jun,Sept,Dec',9:'PromoInterval=Mar,Jun,Sept,Dec',12:'PromoInterval=Mar,Jun,Sept,Dec'}
  if row['Promo2']==1 and row[months[m]]==1:
    if ((row['Date'].year < row['Promo2SinceYear']) or (row['Date'].week > row['Promo2SinceWeek'])):
      return 1
  return 0


def GBModel2(train,test,splitcriteria):
  trains=splitModels(train,splitcriteria)
  tests=splitModels(test,splitcriteria)
  print('starting Gradient Boosting ...')
  print(splitcriteria)
  models=[]
  params=''
  preds=[]
  inp=[]
  # reference
  #GradientBoostingRegressor(n_estimators=350, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7)
  for train,test in zip(trains,tests):
    model=GradientBoostingRegressor(n_estimators=150,max_features='auto',min_samples_split=7,min_samples_leaf=7,max_depth=9,verbose=1)
    trA_X=train.drop('LogSales',axis=1)
    trA_Y=train['LogSales']
    model.fit(trA_X,trA_Y)
    preds.append(getDF(model.predict(test.drop('Id',axis=1))))
    inp.append(test['Id'])
    params=model.get_params()
  
  print('completed Gradient Boosting ...')
  st=strftime("%a, %d %b %Y %H:%M:%S").translate(str.maketrans(' :,','___'))
  fname='Inputs'+st+'.csv'
  pd.concat(inp).to_csv(fname)
  pd.DataFrame([params]).to_csv(fname,mode='a')
  pd.concat(preds).to_csv('Predict_'+fname)

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
  tr_X=train.drop('LogSales',axis=1)
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
print(dtrain.columns)
print(dtest.columns)
#GBModel(dtrain,dtest,gStoreType)
GBModel2(dtrain,dtest,gStoreTypes)
#ridge=RidgeCVLinear(dtrain,dtest)
