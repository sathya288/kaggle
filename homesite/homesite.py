'''

This is a new version to attempt the Kaggle Homesite prediction competition.

__author__ = Sathya, AR
__version__ = 3.14
__date__ = 14th Dec 2015

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import zipfile
from time import time 
from time import strftime
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter
import xgboost as xgb

trainfile='train.csv'
testfile='test.csv'
gNoSplit=[]
keyTrainCols=['QuoteConversion_Flag','CoverageField6A', 'GeographicField41B', 'PropertyField16B', 'GeographicField8A', 'GeographicField1B', 'PropertyField2B', 'GeographicField1A', 'PersonalField17', 'PersonalField16', 'Field8', 'CoverageField6B', 'CoverageField8', 'PropertyField1A', 'SalesField2A', 'PropertyField39B', 'PropertyField1B', 'SalesField8', 'PersonalField11', 'PersonalField27', 'PersonalField15', 'SalesField3', 'PropertyField39A', 'PersonalField84', 'PropertyField34', 'CoverageField11B', 'LogSalesField8', 'CoverageField11A', 'SalesField10', 'PersonalField4A', 'CoverageField9', 'SalesField6', 'PersonalField4B', 'SalesField2B', 'SalesField1A', 'PersonalField13', 'Field7', 'PropertyField29', 'SalesField1B', 'SalesField4', 'PersonalField9', 'PersonalField12', 'PersonalField2', 'PersonalField10A', 'SalesField5', 'PersonalField1', 'PersonalField10B', 'PropertyField37']
keyTestCols=['QuoteNumber','CoverageField6A', 'GeographicField41B', 'PropertyField16B', 'GeographicField8A', 'GeographicField1B', 'PropertyField2B', 'GeographicField1A', 'PersonalField17', 'PersonalField16', 'Field8', 'CoverageField6B', 'CoverageField8', 'PropertyField1A', 'SalesField2A', 'PropertyField39B', 'PropertyField1B', 'SalesField8', 'PersonalField11', 'PersonalField27', 'PersonalField15', 'SalesField3', 'PropertyField39A', 'PersonalField84', 'PropertyField34', 'CoverageField11B', 'LogSalesField8', 'CoverageField11A', 'SalesField10', 'PersonalField4A', 'CoverageField9', 'SalesField6', 'PersonalField4B', 'SalesField2B', 'SalesField1A', 'PersonalField13', 'Field7', 'PropertyField29', 'SalesField1B', 'SalesField4', 'PersonalField9', 'PersonalField12', 'PersonalField2', 'PersonalField10A', 'SalesField5', 'PersonalField1', 'PersonalField10B', 'PropertyField37']

def loadData(trainfile,testfile):
  print('loading training data ...')
  train=pd.read_csv(open(trainfile),parse_dates=['Original_Quote_Date'])
  print('loading testing data ...')
  test=pd.read_csv(open(testfile),parse_dates=['Original_Quote_Date'])
  print('data loading completed...')
  return train,test

def sanitizeData(train):
  print('sanitizing data ...')
  #let us split the date into Month, Day, Year.
  train['Year']=train['Original_Quote_Date'].apply(lambda x: x.year)
  train['Month']=train['Original_Quote_Date'].apply(lambda x: x.month)
  train['Day']=train['Original_Quote_Date'].apply(lambda x: x.day)
  train['DayOfWeek']=train['Original_Quote_Date'].apply(lambda x:x.dayofweek)
  train['DayOfYear']=train['Original_Quote_Date'].apply(lambda x:x.dayofyear)
  #This field has max/min ratio greater than 1000. So lets take the log values
  train['LogSalesField8']=train['SalesField8'].apply(lambda x:math.log(x))
  train.drop(['Original_Quote_Date','SalesField8'],axis=1,inplace=True)
  train= ImputeValues(train)
  return train
  #this is the best we could do.

def ImputeValues(train):
  #Some columns have NaNs
  #If more than 50% of the total has NaN then we drop the column.
  #Else use the most_frequent strategy.
  #Columns to be dropped
  #dropcols=['PersonalField7','PropertyField3','PropertyField29','PropertyField4','PropertyField32','PropertyField34','PropertyField36','PropertyField38']
  #train.drop(dropcols,axis=1,inplace=True)

  #Impute
  #Refer also to https://www.kaggle.com/c/homesite-quote-conversion/forums/t/17417/missing-values
  train.fillna(-1,inplace=True)
  return train

def encode_labels(train,test):
  for f in train.columns:
    if train[f].dtype=='object':
      lbl = preprocessing.LabelEncoder()
      lbl.fit(np.unique(list(train[f].unique()) + list(test[f].unique())))
      train[f] = lbl.transform(list(train[f].values))
      test[f] = lbl.transform(list(test[f].values))
  return train, test

def splitModels(train,cond):
  #split based on storeType
  print('splitting models ...')
  if len(cond)==0:
    return [train]
  if type(cond) is dict:
    for key, val in zip(cond.keys(),cond.values()):
      print([train[train[key]==i].index.size for i in val])
      return [train[train[key]==i] for i in val]
  return [train[train[x]==1] for x in cond]


def test_ModelXY(train,splitcriteria,modelclass,params):
  trains=splitModels(train,splitcriteria)
  preds=[]
  models=[]
  for train in trains:
    model=globals()[modelclass]()
    model.set_params(**params)
    cutoff=math.floor(train.index.size*0.7)
    tr=train[:cutoff]
    te=train[cutoff:]
    model.fit(tr.drop('QuoteConversion_Flag',axis=1),tr['QuoteConversion_Flag'])
    yhat=model.predict_proba(te.drop('QuoteConversion_Flag',axis=1))
    auc=roc_auc_score(te['QuoteConversion_Flag'], pd.DataFrame(yhat).iloc[:,1])
    print('Auc is %6f'%auc)
    preds.append(yhat)
    models.append(model)
  return models

def ModelXY(dtrain,dtest,splitcriteria,modelclass,params):
  trains=splitModels(dtrain,splitcriteria)
  tests=splitModels(dtest,splitcriteria)
  preds=[]
  for train,test in zip(trains,tests):
    model=globals()[modelclass]()
    model.set_params(**params)
    model.fit(train.drop('QuoteConversion_Flag',axis=1),train['QuoteConversion_Flag'])
    yhat=model.predict_proba(test.drop('QuoteNumber',axis=1))
    preds.append(yhat)
  yhats=[hat for sublist in preds for hat in sublist]
  st=strftime("%a, %d %b %Y %H:%M:%S").translate(str.maketrans(' :,','___'))
  fname='Output'+st+'.csv'
  pd.DataFrame(dtest['QuoteNumber']).join(pd.DataFrame(yhats),lsuffix='x').to_csv(fname)
  return None

def XGBModelXY(dtrain,dtest,splitcriteria,modelparams,iters):
  trains=splitModels(dtrain,splitcriteria)
  tests=splitModels(dtest,splitcriteria)
  preds=[]
  for train,test in zip(trains,tests):
    x=train.drop('QuoteConversion_Flag',axis=1)
    y=train['QuoteConversion_Flag']
    clf=xgb.XGBClassifier()
    #gbm=xgb.train(modelparams,xtrain,iters)
    gbm=clf.fit(x,y,eval_metric='auc')
    yhat=gbm.predict_proba(xgb.DMatrix(np.array(test.drop('QuoteNumber',axis=1))))
    preds.append(yhat)
  yhats=[hat for sublist in preds for hat in sublist]
  st=strftime("%a, %d %b %Y %H:%M:%S").translate(str.maketrans(' :,','___'))
  fname='Output'+st+'.csv'
  pd.DataFrame(dtest['QuoteNumber']).join(pd.DataFrame(yhats),lsuffix='x').to_csv(fname)
  return None


def plotFeatureImportance(clf,df,suffix='x'):
  ###############################################################################
  # Plot feature importance
  feature_importance = clf.feature_importances_
  # make importances relative to max importance
  feature_importance = 100.0 * (feature_importance / feature_importance.max())
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + 2.5
  plt.subplot(1, 2, 2)
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, df.columns[sorted_idx])
  plt.xlabel('Relative Importance')
  plt.title('Variable Importance Rossman Dataset')
  ffile='variableImp'+suffix+'.png'
  plt.savefig(ffile)
  plt.clf()

def getReducedSet(df,keyCols):
  return df[keyCols]

def tuneHyperParams(train,modelclass,param_dist,iterations):
  print('Start tuning ... model class ', modelclass)
  model=globals()[modelclass]()
  trA_X=train.drop('QuoteConversion_Flag',axis=1)
  trA_Y=train['QuoteConversion_Flag']
  # run randomized search
  random_search = RandomizedSearchCV(model, param_distributions=param_dist,n_iter=iterations)
  start = time()
  random_search.fit(trA_X, trA_Y)
  print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time() - start), iterations))
  report(random_search.grid_scores_)

# Utility function to report best scores
def report(grid_scores, n_top=3):
  top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
  for i, score in enumerate(top_scores):
    print("Model with rank: {0}".format(i + 1))
    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score, np.std(score.cv_validation_scores)))
    print("Parameters: {0}".format(score.parameters))
    print("")

#split features based on the category
def splitCategories(columns):
  covmodel=[]
  salesmodel=[]
  persmodel=[]
  propmodel=[]
  geomodel=[]
  genmodel=[]
  for col in colsT:
    lead=col[:4]
    if lead=='Cove':
      covmodel.append(col)
    elif lead=='Sale':
      salesmodel.append(col)
    elif lead=='Pers':
      persmodel.append(col)
    elif lead=='Prop':
      propmodel.append(col)
    elif lead=='Geog':
      geomodel.append(col)
    else:
      genmodel.append(col)
  return covmodel,salesmodel,persmodel,propmodel,geomodel,genmodel


train,test=loadData(trainfile,testfile)
train.drop('QuoteNumber',axis=1,inplace=True)
dtrain=sanitizeData(train)
dtest=sanitizeData(test)
dtrain,dtest=encode_labels(dtrain,dtest)
print('get reduced set')
#dtrain=getReducedSet(dtrain,keyTrainCols)
#dtest=getReducedSet(dtest,keyTestCols)
print('got reduced set')
splitcriteria=[]
params={'n_estimators':1000,'max_features':'auto','max_depth':8,'min_samples_leaf':8,'min_samples_split':10,'verbose':1}
modelclass='GradientBoostingClassifier'
#modelclass='RandomForestClassifier'
#clf=test_ModelXY(dtrain,gNoSplit,modelclass,params)
#plotFeatureImportance(clf[0],dtrain)
#ModelXY(dtrain,dtest,gNoSplit,modelclass,params)
params_dist = {"max_depth":sp_randint(1,12),
              "max_features": ['log2','auto','sqrt'],
              "min_samples_split": sp_randint(5, 15),
              "min_samples_leaf": sp_randint(5, 15),
              "learning_rate": [0.001,0.01,0.05,0.025,0.075,0.1], "warm_start":True}
itr=1
xg_params={'bst:max_depth':3, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse' }
iters=3

res=XGBModelXY(dtrain,dtest,gNoSplit,xg_params,iters)
#tuneHyperParams(dtrain,'GradientBoostingClassifier',params_dist,itr)
