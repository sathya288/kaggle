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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RidgeCV
import zipfile
from time import time 
from time import strftime
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn import cross_validation

gStoreTypes=['StoreType=a','StoreType=b','StoreType=c','StoreType=d']
gPromoInterval=['PromoInterval=0', 'PromoInterval=Feb,May,Aug,Nov','PromoInterval=Jan,Apr,Jul,Oct', 'PromoInterval=Mar,Jun,Sept,Dec']
gAssortment=['Assortment=a', 'Assortment=b','Assortment=c']
gQuarter={'Quarter':[1,2,3,4]}
gWeekly={'DayOfWeek':[1,2,3,4,5,6,7]}
gMonthly={'Month':[1,2,3,4,5,6,7,8,9,10,11,12]}
gYearly={'Year':[2013,2014,2015]}
gDay={'Day':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]}
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

  perStore=dtrain.groupby('Store')
  pStore=pd.DataFrame()
  pStore['meanperStore']=perStore['LogSales'].mean()
  pStore['sdperStore']=perStore['LogSales'].std()
  pStore['Store']=pStore.index
  pStore['salepercustomerPerStore']=perStore.apply(func=lambda row:row['LogSales']/row['LogCust'])
  
  #Sales per DayofYear, Month,DayofWeek,Year per Store.
  lspdoy=pd.DataFrame()
  lspdoy['lspdoy']=dtrain.groupby(['Store','DayOfYear'])['LogSales'].mean()
  lspdoy['lspdoysd']=dtrain.groupby(['Store','DayOfYear'])['LogSales'].std()
  lspdoy.reset_index(inplace=True)
  lspdow=pd.DataFrame()
  lspdow['lspdow']=dtrain.groupby(['Store','DayOfWeek'])['LogSales'].mean()
  lspdow['lspdowsd']=dtrain.groupby(['Store','DayOfWeek'])['LogSales'].std()
  lspdow.reset_index(inplace=True)
  lspy=pd.DataFrame()
  lspy['lspy']=dtrain.groupby(['Store','Year'])['LogSales'].mean()
  lspy['lspysd']=dtrain.groupby(['Store','Year'])['LogSales'].std()
  lspy.reset_index(inplace=True)
  lspm=pd.DataFrame()
  lspm['lspm']=dtrain.groupby(['Store','Month'])['LogSales'].mean()
  lspm['lspmsd']=dtrain.groupby(['Store','Month'])['LogSales'].std()
  lspm.reset_index(inplace=True)
  
  lcpdoy=pd.DataFrame()
  lcpdoy['lcpdoy']=dtrain.groupby(['Store','DayOfYear'])['LogCust'].mean()
  lcpdoy['lcpdoysd']=dtrain.groupby(['Store','DayOfYear'])['LogCust'].std()
  lcpdoy.reset_index(inplace=True)
  lcpdow=pd.DataFrame()
  lcpdow['lcpdow']=dtrain.groupby(['Store','DayOfWeek'])['LogCust'].mean()
  lcpdow['lcpdowsd']=dtrain.groupby(['Store','DayOfWeek'])['LogCust'].std()
  lcpdow.reset_index(inplace=True)
  lcpy=pd.DataFrame()
  lcpy['lcpy']=dtrain.groupby(['Store','Year'])['LogCust'].mean()
  lcpy['lcpysd']=dtrain.groupby(['Store','Year'])['LogCust'].std()
  lcpy.reset_index(inplace=True)
  lcpm=pd.DataFrame()
  lcpm['lcpm']=dtrain.groupby(['Store','Month'])['LogCust'].mean()
  lcpm['lcpmsd']=dtrain.groupby(['Store','Month'])['LogCust'].std()
  lcpm.reset_index(inplace=True)

  #merge these newly calculated perStore statistics
  dtrain = pd.merge(dtrain,pStore,how='inner',on='Store')
  dtrain = pd.merge(dtrain,lspdoy,how='left',on=['Store','DayOfYear'])
  dtrain = pd.merge(dtrain,lspdow,how='left',on=['Store','DayOfWeek'])
  dtrain = pd.merge(dtrain,lspy,how='left',on=['Store','Year'])
  dtrain = pd.merge(dtrain,lspm,how='left',on=['Store','Month'])
  dtrain = pd.merge(dtrain,lcpdoy,how='left',on=['Store','DayOfYear'])
  dtrain = pd.merge(dtrain,lcpdow,how='left',on=['Store','DayOfWeek'])
  dtrain = pd.merge(dtrain,lcpy,how='left',on=['Store','Year'])
  dtrain = pd.merge(dtrain,lcpm,how='left',on=['Store','Month'])
  dtrain['lspdoy'].fillna(0,inplace=True)
  dtrain['lspdoysd'].fillna(0,inplace=True)
  dtrain['lspdow'].fillna(0,inplace=True)
  dtrain['lspdowsd'].fillna(0,inplace=True)
  dtrain['lspy'].fillna(0,inplace=True)
  dtrain['lspysd'].fillna(0,inplace=True)
  dtrain['lspm'].fillna(0,inplace=True)
  dtrain['lspmsd'].fillna(0,inplace=True)
  dtrain['lcpdoy'].fillna(0,inplace=True)
  dtrain['lcpdoysd'].fillna(0,inplace=True)
  dtrain['lcpdow'].fillna(0,inplace=True)
  dtrain['lcpdowsd'].fillna(0,inplace=True)
  dtrain['lcpy'].fillna(0,inplace=True)
  dtrain['lcpysd'].fillna(0,inplace=True)
  dtrain['lcpm'].fillna(0,inplace=True)
  dtrain['lcpmsd'].fillna(0,inplace=True)
  
  dtest= pd.merge(dtest,pStore,how='inner',on='Store')
  dtest = pd.merge(dtest,lspdoy,how='left',on=['Store','DayOfYear'])
  dtest = pd.merge(dtest,lspdow,how='left',on=['Store','DayOfWeek'])
  dtest = pd.merge(dtest,lspy,how='left',on=['Store','Year'])
  dtest = pd.merge(dtest,lspm,how='left',on=['Store','Month'])
  dtest = pd.merge(dtest,lcpdoy,how='left',on=['Store','DayOfYear'])
  dtest = pd.merge(dtest,lcpdow,how='left',on=['Store','DayOfWeek'])
  dtest = pd.merge(dtest,lcpy,how='left',on=['Store','Year'])
  dtest = pd.merge(dtest,lcpm,how='left',on=['Store','Month'])

  dtest['lspdoy'].fillna(0,inplace=True)
  dtest['lspdoysd'].fillna(0,inplace=True)
  dtest['lspdow'].fillna(0,inplace=True)
  dtest['lspdowsd'].fillna(0,inplace=True)
  dtest['lspy'].fillna(0,inplace=True)
  dtest['lspysd'].fillna(0,inplace=True)
  dtest['lspm'].fillna(0,inplace=True)
  dtest['lspmsd'].fillna(0,inplace=True)
  dtest['lcpdoy'].fillna(0,inplace=True)
  dtest['lcpdoysd'].fillna(0,inplace=True)
  dtest['lcpdow'].fillna(0,inplace=True)
  dtest['lcpdowsd'].fillna(0,inplace=True)
  dtest['lcpy'].fillna(0,inplace=True)
  dtest['lcpysd'].fillna(0,inplace=True)
  dtest['lcpm'].fillna(0,inplace=True)
  dtest['lcpmsd'].fillna(0,inplace=True)

  print('sanitizing data ... completed')
  print('feature engg ...')
  #columns to be dropped based on simple correlation
  #dropcols=['Promo2', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval','Date','Open','StateHoliday','SchoolHoliday','Assortment','CompetitionDistance']
  #The above two lines were from previous run which fetched the highest score till date. So copying here for reference.
  #dtrain['IsPromo2On']=dtrain.apply(func=getPromo2,axis=1)
  #dtest['IsPromo2On']=dtest.apply(func=getPromo2,axis=1)
  dtrain['Quarter']=dtrain.apply(func=getQuarter,axis=1)
  dtest['Quarter']=dtest.apply(func=getQuarter,axis=1)
  #dtrain['SalesPerCustomer']=dtrain.apply(func=getSPC,axis=1)
  #dropcols=['Open','Date']
  #dropcols=['Promo2', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','Date','Open','StateHoliday','SchoolHoliday','CompetitionDistance']
  dropcols=['Promo2', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','Date','Open','StateHoliday','SchoolHoliday','CompetitionDistance']
  traindropcols=['Sales','Customers']
  dtrain.drop(traindropcols,axis=1,inplace=True)
  dtrain.drop(dropcols,axis=1,inplace=True)
  dtest.drop(dropcols,axis=1,inplace=True)
  dtrain.drop(gPromoInterval,axis=1,inplace=True)
  dtest.drop(gPromoInterval,axis=1,inplace=True)
  dtrain.drop(gAssortment,axis=1,inplace=True)
  dtest.drop(gAssortment,axis=1,inplace=True)
  #dtrain.drop(gStoreTypes,axis=1,inplace=True)
  #dtest.drop(gStoreTypes,axis=1,inplace=True)
  print('feature engg ... completed')
  #return encode_onehot(dtrain,['IsPromo2On']), encode_onehot(dtest,['IsPromo2On'])
  return dtrain,dtest

def getSPC(row):
  sales=math.floor(math.exp(row['LogSales']))
  customers=row['Customers']
  return math.floor(sales/customers)

def sanitizeInputs(train):
  print('sanitizing Training data ... ')
  train['Open'].replace('NaN',1,inplace=True)
  train['Day']=train['Date'].apply(lambda x:x.day)
  train['Month']=train['Date'].apply(lambda x:x.month)
  train['Year']=train['Date'].apply(lambda x:x.year)
  train['DayOfYear']=train['Date'].apply(lambda x:x.dayofyear)
  if 'Sales' in train.columns:
    train=train[train['Open']>0]
    train=train[train['Sales']>0]
    train.loc[:,'LogSales']=train['Sales'].apply(lambda x:1000*math.log(x+1))
    train.loc[:,'LogCust']=train['Customers'].apply(lambda x:100*math.log(x+1))
  train['StateHoliday'].replace({'0':0,'a':1,'b':2,'c':3},inplace=True)
  print('sanitizing Training data ... completed')
  return encode_onehot(train,['DayOfWeek','Day','Month','Year','Store','SchoolHoliday','StateHoliday'])

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
  store['LogCompDist']=store['CompetitionDistance'].apply(lambda x:math.log(x+1))
  store['LogCompDays']=store.apply(func=getDays,axis=1)
  print('sanitizing Store data ... completed')
  return encode_onehot(store,['Store','StoreType','Assortment','PromoInterval'])
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

def getPromo2(row):
  m=row['Date'].month
  months={0:'PromoInterval=0', 1:'PromoInterval=Jan,Apr,Jul,Oct',4:'PromoInterval=Jan,Apr,Jul,Oct',7:'PromoInterval=Jan,Apr,Jul,Oct',10:'PromoInterval=Jan,Apr,Jul,Oct',2:'PromoInterval=Feb,May,Aug,Nov',5:'PromoInterval=Feb,May,Aug,Nov',8:'PromoInterval=Feb,May,Aug,Nov',11:'PromoInterval=Feb,May,Aug,Nov',3:'PromoInterval=Mar,Jun,Sept,Dec',6:'PromoInterval=Mar,Jun,Sept,Dec',9:'PromoInterval=Mar,Jun,Sept,Dec',12:'PromoInterval=Mar,Jun,Sept,Dec'}
  if row['Promo2']==1 and row[months[m]]==1:
    if ((row['Date'].year < row['Promo2SinceYear']) or (row['Date'].week > row['Promo2SinceWeek'])):
      return 1
  return 0

def getTrainedModel(X,Y,model):
  scores=cross_validation.cross_val_score(model,X,Y,scoring=myScore,cv=5,n_jobs=-1,verbose=1)
  print('Mean RMSPE ', scores.mean())
  return model, scores.mean()

def myScore(clf,X,Y):
  yhat=clf.predict(X)
  return rmspe(yhat,Y)


def GBModel2(train,test,splitcriteria,modelclass,modelparams,colname):
  train=train.drop('LogCust',axis=1)
  trains=splitModels(train,splitcriteria)
  tests=splitModels(test,splitcriteria)
  print('starting Gradient Boosting ...')
  models=[]
  params=''
  preds=[]
  inp=[]
  cond=''
  # reference
  #GradientBoostingRegressor(n_estimators=350, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7)
  for train,test in zip(trains,tests):
    if test.index.size >0 and train.index.size>0:
      model=globals()[modelclass](n_estimators=500)
      model.set_params(**modelparams)
      trA_X=train.drop('LogSales',axis=1)
      trA_Y=train['LogSales']
      model,score=getTrainedModel(trA_X,trA_Y,model)
      model.fit(trA_X,trA_Y)
      why=test.drop('Id',axis=1)
      yhat=model.predict(why)
      preds.append(getDF(yhat))
      inp.append(test['Id'])
      params=model.get_params()
    else:
      print('Inner condition is ',cond)
      if train.index.size==0:
        print('No Training Data', cond)
      if test.index.size==0:
        print('No Test Data',cond)

  print('completed Gradient Boosting ...')
  st=strftime("%a, %d %b %Y %H:%M:%S").translate(str.maketrans(' :,','___'))
  fname1='Inputs'+ str(math.floor(1000*np.random.rand())) + st +'.csv'
  fname2='Predict'+ str(math.floor(1000*np.random.rand())) + st +'.csv'
  x=np.array(pd.concat(inp))
  y=np.array(pd.concat(preds))
  z=pd.DataFrame(x,columns=['Id'],index=np.arange(len(x)).tolist())
  z[colname]=y
  return z

def MultiGBModel2(dtrain,dtest,gDay,gbmodel,params,stuff,que):
  que.put(GBModel2(dtrain,dtest,gDay,gbmodel,params,stuff))

def XGBModel(train,test,splitcriteria,iters,modelparams,colname):
  train=train.drop('LogCust',axis=1)
  trains=splitModels(train,splitcriteria)
  tests=splitModels(test,splitcriteria)
  print('starting XGradient Boosting ...')
  params=''
  preds=[]
  inp=[]
  # reference
  #GradientBoostingRegressor(n_estimators=350, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7)
  for train,test in zip(trains,tests):
    print('XGB .. ', splitcriteria)
    if test.index.size >0 and train.index.size>0:
      val_idx=math.floor(0.2*train.index.size)
      trA_X=train.drop('LogSales',axis=1)
      trA_Y=train['LogSales']
      xval=xgb.DMatrix(np.array(trA_X[:val_idx]), np.array(trA_Y[:val_idx]))
      xtrain=xgb.DMatrix(np.array(trA_X[val_idx:]), np.array(trA_Y[val_idx:]))
      watchlist=[(xval,'eval'),(xtrain,'train')]
      xtrain2=xgb.DMatrix(np.array(trA_X), np.array(trA_Y))
      gbm=xgb.train(modelparams,xtrain,iters,feval=rmspe_xg)
      why=test.drop('Id',axis=1)
      yhat=gbm.predict(xgb.DMatrix(np.array(why)))
      preds.append(getDF(yhat))
      inp.append(test['Id'])
    else:
      print('XGB Inner condition is ')
      if train.index.size==0:
        print('XGB No Training Data')
      if test.index.size==0:
        print('XGB No Test Data')

  print('completed XGradient Boosting ...')
  st=strftime("%a, %d %b %Y %H:%M:%S").translate(str.maketrans(' :,','___'))
  fname1='XInputs'+ str(math.floor(1000*np.random.rand())) + st +'.csv'
  fname2='XPredict'+ str(math.floor(1000*np.random.rand())) + st +'.csv'
  x=np.array(pd.concat(inp))
  y=np.array(pd.concat(preds))
  z=pd.DataFrame(x,columns=['Id'],index=np.arange(len(x)).tolist())
  z[colname]=y
  return z

def getDF(testY):
  t=[]
  for y in testY:
    t.append(math.exp(y/1000)-1)
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

def rmspe_xg(yhat, y):
  # y = y.values
  y = y.get_label()
  y = np.exp(y) - 1
  yhat = np.exp(yhat) - 1
  w = ToWeight(y)
  mspe = np.sqrt(np.mean(w * (y - yhat)**2))
  return "rmspe", mspe


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
  if type(cond) is dict:
    for key, val in zip(cond.keys(),cond.values()):
      print([train[train[key]==i].index.size for i in val])
      return [train[train[key]==i].drop(key,axis=1)for i in val]
  return [train[train[x]==1].drop(x,axis=1) for x in cond]
  print('splitting models ... completed')

def plotFeatureImportance(clf,df,suffix='x'):
  ###############################################################################
  # Plot feature importance
  feature_importance = clf.feature_importances_
  # make importances relative to max importance
  feature_importance = 100.0 * (feature_importance / feature_importance.max())
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + .5
  plt.subplot(1, 2, 2)
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, df.columns[sorted_idx])
  plt.xlabel('Relative Importance')
  plt.title('Variable Importance Rossman Dataset')
  ffile='variableImp'+suffix+'.png'
  plt.savefig(ffile)
  plt.clf()

def prepareDiagnosticsData(dtrain):
  #prepare validation data so that we can almost mimic leaderboard scores.
  sts=splitModels(dtrain,gStoreTypes)
  tests=[]
  trains=[]
  for st in sts:
    cutoff=math.floor(st.index.size*0.7)
    tests.append(st[cutoff:])
    trains.append(st[:cutoff])
  tst=pd.concat(tests)
  tst['Id']=tst.index
  return pd.concat(trains), tst

def getQuarter(row):
  quarter={1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
  return quarter[row['Month']]

def tuneHyperParams(train,splitcriteria,modelclass,param_dist,iterations):
  trains=splitModels(train,splitcriteria)
  print('Start tuning ... model class ', modelclass)
  print('Split criteria is - ', splitcriteria)
  for train in trains:
    model=globals()[modelclass]()
    trA_X=train.drop('LogSales',axis=1)
    trA_Y=train['LogSales']
    # run randomized search
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,n_iter=iterations)
    start = time()
    random_search.fit(trA_X, trA_Y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), iterations))
    report(random_search.grid_scores_)

def NestedModels(train,test,splitcriteria,modelclass,params):

  tr_Cond=splitcriteria[-1]
  trains=splitModels(train,splitcriteria[0])
  tests=splitModels(test,splitcriteria[0])
  sc=''
  if type(splitcriteria)=='dict':
    for val in splitcriteria[1].values():
      sc=val
  else:
    sc=splitcriteria[0]
  for tr,te,cond in zip(trains,tests,sc):
    print('Outer Split Condition', ' - ',cond)
    if te.index.size >0:
      GBModel2(tr,te,splitcriteria[1],modelclass,params)

# Utility function to report best scores
def report(grid_scores, n_top=3):
  top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
  for i, score in enumerate(top_scores):
    print("Model with rank: {0}".format(i + 1))
    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score, np.std(score.cv_validation_scores)))
    print("Parameters: {0}".format(score.parameters))
    print("")

def prepareValidationData(dtrain):
  #Validation data mimics the test data. So the distribution etc are as close as possible.
  #This is so as to get a realistic validation error.
  #Store types are equal. PromoIntervals are equal. 
  #Only months 8 and 9.
  msamples=pd.concat([dtrain[dtrain['Month']==8],dtrain[dtrain['Month']==9]])
  samples=splitModels(msamples,gStoreTypes)
  dtest=[]
  for sample in samples:
    print('Test data size',sample.index.size)
    samptest=sample.iloc[:math.floor(sample.index.size*.25)]
    samptest['Id']=samptest.index
    dtest.append(samptest)
  ntest=pd.concat(dtest)
  print('Before',dtrain.index.size)
  dtrain=pd.DataFrame(dtrain,index=dtrain.index.difference(ntest.index))
  print('After',dtrain.index.size)
  return dtrain,ntest

def getReducedData(dtrain,dtest):
  #lets limit to the training data based on what is available in test - a dumbastic approach - but lets do it.
  return dtrain[dtrain['Quarter']==3],dtest


def getError(inp,oup,valtest):
  res=pd.concat(inp).join(pd.concat(oup))
  res.sort_index(inplace=True)
  valtest.sort_index(inplace=True)
  if res.index != valtest.index:
    print('something went majorly wrong')
    return -1
  err=rmspe(res.iloc[:,-1].apply(lambda x:math.log(x)),valtest['LogSales'])
  return err

'''
train,test,store=loadData('data/train.csv.zip', 'data/test.csv.zip','data/store.csv.zip')
dtrain,dtest=sanitizeData(train,test,store)
#Let us persist the data so that we can quickly start the next time
dtrain.to_csv('prepTrain.csv')
dtest.to_csv('prepTest.csv')
print('before exit')
exit
print('after exit')
'''
#load from preps
dtrain=pd.read_csv(open('prepTrain.csv'))
dtest=pd.read_csv(open('prepTest.csv'))

#dtrain,dtest=prepareDiagnosticsData(dtrain)
#dtrain,dtest=getReducedData(dtrain,dtest)
params={'n_estimators':700,'max_features':'sqrt','max_depth':11,'min_samples_leaf':8,'min_samples_split':8,'verbose':0,'learning_rate':0.08}
#params={'n_estimators':3,'max_features':'auto','max_depth':9,'min_samples_leaf':8,'min_samples_split':8,'verbose':1,'learning_rate':0.075}
#rf_params={'n_estimators':600,'max_features':'auto','max_depth':22,'min_samples_leaf':6,'min_samples_split':6,'verbose':1,'n_jobs':-1}
#gbmodel='RandomForestRegressor'
gbmodel='GradientBoostingRegressor'
#gbmodel='GradientBoostingRegressor'
#dtrain,valtest= prepareValidationData(dtrain)
#dtest=valtest.drop('LogSales',axis=1)
#NestedModels(dtrain,dtest,[gPromoInterval,gStoreTypes],gbmodel,params)

#nosplit_params= [{'max_depth': , 'max_features': , 'learning_rate': , 'min_samples_leaf': ,'min_samples_split':}]
#params=rf_params
res=GBModel2(dtrain,dtest,gSingle,gbmodel,params,'nosp')
res=res.merge(GBModel2(dtrain,dtest,gDay,gbmodel,params,'day'),on='Id',sort=True)
#res=res.merge(GBModel2(dtrain,dtest,gQuarter,gbmodel,params,'quart'),on='Id',sort=True)
#res=res.merge(GBModel2(dtrain,dtest,gMonthly,gbmodel,params,'month'),on='Id',sort=True)
res=res.merge(GBModel2(dtrain,dtest,gWeekly,gbmodel,params,'week'),on='Id',sort=True)
#res=res.merge(GBModel2(dtrain,dtest,gYearly,gbmodel,params,'year'),on='Id',sort=True)
res=res.merge(GBModel2(dtrain,dtest,gStoreTypes,gbmodel,params,'store'),on='Id',sort=True)
#res=res.merge(GBModel2(dtrain,dtest,gPromoInterval,gbmodel,params,'promo'),on='Id',sort=True)
#res=res.merge(GBModel2(dtrain,dtest,gAssortment,gbmodel,params,'assort'),on='Id',sort=True)

st=strftime("%a, %d %b %Y %H:%M:%S").translate(str.maketrans(' :,','___'))
fname1='GBMPredict'+ st +'.csv'
res.to_csv(fname1)
#print(getError(inp,oup,valtest))
#EnsemblePrediction()
#ridge=RidgeCVLinear(dtrain,dtest)
# specify parameters and distributions to sample from
'''
xg_params={'max_depth':10, 'eta':0.1,'colsample_bytree':0.8,'subsample':0.7,'silent':2, 'objective':'reg:linear','booster':'gbtree','eval_metric':'rmse'}
iters=2400
res=XGBModel(dtrain,dtest,gSingle,iters,xg_params,'nosp')
'''
'''
res=res.merge(XGBModel(dtrain,dtest,gDay,iters,xg_params,'day'),on='Id',sort=True,suffixes=('_1','_2'))
res=res.merge(XGBModel(dtrain,dtest,gQuarter,iters,xg_params,'quart'),on='Id',sort=True,suffixes=('_1','_2'))
res=res.merge(XGBModel(dtrain,dtest,gMonthly,iters,xg_params,'month'),on='Id',sort=True,suffixes=('_1','_2'))
res=res.merge(XGBModel(dtrain,dtest,gWeekly,iters,xg_params,'week'),on='Id',sort=True,suffixes=('_1','_2'))
res=res.merge(XGBModel(dtrain,dtest,gYearly,iters,xg_params,'year'),on='Id',sort=True,suffixes=('_1','_2'))
res=res.merge(XGBModel(dtrain,dtest,gStoreTypes,iters,xg_params,'store'),on='Id',sort=True,suffixes=('_1','_2'))
res=res.merge(XGBModel(dtrain,dtest,gPromoInterval,iters,xg_params,'promo'),on='Id',sort=True,suffixes=('_1','_2'))
res=res.merge(XGBModel(dtrain,dtest,gAssortment,iters,xg_params,'assort'),on='Id',sort=True,suffixes=('_1','_2'))
st=strftime("%a, %d %b %Y %H:%M:%S").translate(str.maketrans(' :,','___'))
fname1='XGB_Predict'+ st +'.csv'
res.to_csv(fname1)
'''
#params_dist = {"max_depth":sp_randint(1,12), "max_features": ['log2','auto','sqrt'], "min_samples_split": sp_randint(5, 15), "min_samples_leaf": sp_randint(5, 15), "learning_rate": [0.001,0.01,0.05,0.025,0.075,0.1]}
#itr=50
#tuneHyperParams(dtrain,gStoreTypes,gbmodel,params_dist,itr)
#tuneHyperParams(dtrain,gPromoInterval,gbmodel,params_dist,itr)
#tuneHyperParams(dtrain,gAssortment,gbmodel,params_dist,itr)
#tuneHyperParams(dtrain,gQuarter,gbmodel,params_dist,itr)
#tuneHyperParams(dtrain,gMonthly,gbmodel,params_dist,itr)
#tuneHyperParams(dtrain,gWeekly,gbmodel,params_dist,itr)
#tuneHyperParams(dtrain,gSingle,gbmodel,params_dist,itr)
storetype_params=[{'max_depth':3 , 'max_features':'auto' , 'learning_rate':0.05 , 'min_samples_leaf':8 ,'min_samples_split':5}, {'max_depth':2 , 'max_features':'log2' , 'learning_rate':0.1 , 'min_samples_leaf':13 ,'min_samples_split':10}, {'max_depth':1 , 'max_features':'sqrt' , 'learning_rate':0.075 , 'min_samples_leaf':6 ,'min_samples_split':11}, {'max_depth':10 , 'max_features':'log2' , 'learning_rate':0.025 , 'min_samples_leaf':7 ,'min_samples_split':7}]

promoint_params=[ {'max_depth': 5, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 10,'min_samples_split':14 }, {'max_depth': 4, 'max_features': 'log2', 'learning_rate': 0.025, 'min_samples_leaf': 7,'min_samples_split':9 }, {'max_depth': 2, 'max_features': 'sqrt', 'learning_rate': 0.1, 'min_samples_leaf': 14,'min_samples_split':11 }, {'max_depth': 3, 'max_features': 'auto', 'learning_rate': 0.1, 'min_samples_leaf': 14,'min_samples_split':6 }]

assortment_params=[ {'max_depth': 3, 'max_features': 'sqrt', 'learning_rate': 0.075, 'min_samples_leaf': 14,'min_samples_split':6 }, {'max_depth': 6, 'max_features': 'sqrt', 'learning_rate': 0.1, 'min_samples_leaf': 13,'min_samples_split':13 }, {'max_depth': 6, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 13,'min_samples_split':13 }]

quarter_params= [ {'max_depth': 5, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 7,'min_samples_split':7 }, {'max_depth': 6, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 11,'min_samples_split':9 }, {'max_depth': 4, 'max_features': 'auto', 'learning_rate': 0.05, 'min_samples_leaf': 5,'min_samples_split':12 }, {'max_depth': 8, 'max_features': 'auto', 'learning_rate': 0.025, 'min_samples_leaf': 7,'min_samples_split':6 }]

monthly_params= [{'max_depth': 5, 'max_features': 'log2', 'learning_rate': 0.05, 'min_samples_leaf': 11,'min_samples_split':6 }, {'max_depth': 5, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 5,'min_samples_split':8 }, {'max_depth': 4, 'max_features': 'auto', 'learning_rate': 0.05, 'min_samples_leaf': 5,'min_samples_split':9 }, {'max_depth': 6, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 6,'min_samples_split':14 }, {'max_depth': 6, 'max_features': 'log2', 'learning_rate': 0.05, 'min_samples_leaf': 10,'min_samples_split':9 }, {'max_depth': 4, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 11,'min_samples_split':9 }, {'max_depth': 5, 'max_features': 'log2', 'learning_rate': 0.05, 'min_samples_leaf': 9,'min_samples_split':6 }, {'max_depth': 5, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 9,'min_samples_split':6 }, {'max_depth': 6, 'max_features': 'auto', 'learning_rate': 0.025, 'min_samples_leaf': 11,'min_samples_split':6 }, {'max_depth': 6, 'max_features': 'log2', 'learning_rate': 0.05, 'min_samples_leaf': 8,'min_samples_split':11 }, {'max_depth': 5, 'max_features': 'log2', 'learning_rate': 0.05, 'min_samples_leaf': 5,'min_samples_split':10 }, {'max_depth': 5, 'max_features': 'auto', 'learning_rate': 0.05, 'min_samples_leaf': 8,'min_samples_split':12 }]
weekly_params= [{'max_depth': 4, 'max_features': 'auto', 'learning_rate': 0.075, 'min_samples_leaf': 13,'min_samples_split':8 }, {'max_depth': 4, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 7,'min_samples_split':6 }, {'max_depth': 3, 'max_features': 'log2', 'learning_rate': 0.1, 'min_samples_leaf': 14,'min_samples_split':13 }, {'max_depth': 3, 'max_features': 'sqrt', 'learning_rate': 0.1, 'min_samples_leaf': 9,'min_samples_split':11 }, {'max_depth': 3, 'max_features': 'log2', 'learning_rate': 0.1, 'min_samples_leaf': 8,'min_samples_split':9 }, {'max_depth': 5, 'max_features': 'sqrt', 'learning_rate': 0.05, 'min_samples_leaf': 12,'min_samples_split':10 }, {'max_depth': 2, 'max_features': 'auto', 'learning_rate': 0.1, 'min_samples_leaf': 6,'min_samples_split':12 }]
