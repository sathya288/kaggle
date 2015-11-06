'''
Parse the Rossmann data files and prepare the data amenable for analysis
Perform prediction using some models and compute their rmspe
'''

__author__='Sathya AR'
__version__='3.4'
__date__='22/10/2015'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LassoLarsCV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
import seaborn as sns
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import Imputer
from sklearn.cluster import Birch
import matplotlib.colors as colors
from itertools import cycle
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFE


def prepData(trainingfile, storefile,csvfile):
  #load the files
  z = zipfile.ZipFile(trainingfile)
  data=pd.read_csv(z.open(csvfile), parse_dates=['Date'],low_memory=False)
  data.dropna(inplace=True)
  #let us correct the data
  if 'Sales' in data.columns:
    data = data[data['Sales'] >0]
    #let us take the log of Sales in order to make it smooth
    data['logSales']=data['Sales'].apply(lambda x: math.log(x))
    data.drop('Sales',axis=1,inplace=True)
  #load the store. merge it with the train data
  z=zipfile.ZipFile(storefile)
  store=pd.read_csv(z.open('store.csv'))

  data= pd.merge(data,store,how='outer', on='Store')

  #specify all the categorical variables
  data['Year']=data['Date'].apply(lambda x:x.year)
  data['Month']=data['Date'].apply(lambda x:x.month)
  data['Day']=data['Date'].apply(lambda x:x.day)
  #data['IsPromo2On']=data.apply(func=getPromo2,axis=1)
  #data['Promo2SinceWeek']=data.apply(func=getAttrP2SW, axis=1)
  #data['Promo2SinceYear']=data.apply(func=getAttrP2SY, axis=1)
  #data['PromoInterval']=data.apply(func=getAttrP2I, axis=1)
  #Calculate days
  data['CompetitionOpenSinceMonth'].replace('NaN', 0, inplace=True)
  data['CompetitionOpenSinceYear'].replace('NaN', 0, inplace=True)
  data['CompetitionDistance'].replace('NaN', 1, inplace=True)
  data['CompetitionDistance'].replace('NaN', 1, inplace=True)
  data['Open'].replace('NaN', 1, inplace=True)
  data['StateHoliday'].replace('NaN',0,inplace=True)
  data['StateHoliday'].replace('a',1,inplace=True)
  data['StateHoliday'].replace('b',2,inplace=True)
  data['StateHoliday'].replace('c',3,inplace=True)
  
  #Imputer(missing_values='NaN',strategy='most_frequent',copy=False).fit_transform(data['CompetitionOpenSinceMonth'])
  #Imputer(missing_value='NaN', strategy='most_frequent',copy=False).fit_transform(data['CompetitionOpenSinceYear'])
  #Imputer(missing_value='NaN', strategy='most_frequent',copy=False).fit_transform(data['CompetitionDistance'])
   
  data['logCompDist']=data['CompetitionDistance'].apply(lambda x: math.log(x))

  data['CompetitionDays']=data.apply(func=getDays, axis=1)
  #specify all the categorical variables
  catcols=['Store','Open','Day','DayOfWeek','Promo','StoreType','Promo2']
  # Lets use oneHot encoder now.
  data =encode_onehot(data, catcols)
  # Let us ignore the following columns
  dropcols=['Promo2SinceYear','Date','CompetitionDistance','Month','PromoInterval','CompetitionOpenSinceYear','CompetitionOpenSinceMonth','CompetitionDays', 'Year','Promo2SinceYear','SchoolHoliday','Assortment']
  #dropcols=['Date','CompetitionDistance','StateHoliday']
  data.drop(dropcols,axis=1,inplace=True)
  if 'Customers' in data.columns:
    data.drop('Customers',axis=1,inplace=True)

  print(data.columns)
  return data

def getAttrP2SW(row):
  if row['Promo2'] == 0:
    return 0
  return row['Promo2SinceWeek']

def getAttrP2SY(row):
  if row['Promo2'] == 0:
    return 0
  return row['Promo2SinceYear']

def getAttrP2I(row):
  if row['Promo2'] == 0:
    return 0
  return row['PromoInterval']

def getPromo2(row):
  m=row['Date'].month
  months={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
  if row['Promo2']==1 and months[m] in row['PromoInterval']: 
    if row['Date'].year < row['Promo2SinceYear']:
      return 1
    elif row['Date'].week > row['Promo2SinceWeek']:
      return 1
  return 0
      

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
  vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
  vec_data.columns = vec.get_feature_names()
  vec_data.index = df.index
																		    
  df = df.drop(cols, axis=1)
  df = df.join(vec_data)
  return df

def getDays(row):
  month=row['CompetitionOpenSinceMonth']
  year=row['CompetitionOpenSinceYear']
  if (month==0 or year==0):
    return 0
  #approx days will do.
  return math.log((12-month)*30 + (2016-year)*12*30)

def BuildModel(model,df,suffix='x'):
  #shuffle the dataset
  df=df.reindex(np.random.permutation(df.index))
  tr_Y=df['logSales']
  tr_X=df.drop('logSales',axis=1)
  #kf = cross_validation.KFold(tr_Y.size, n_folds=10)
  cutoff=tr_Y.size 
  cutoff=math.floor(0.8*cutoff)
  trainX=tr_X.iloc[0:cutoff]
  trainY=tr_Y.iloc[0:cutoff]
  testX=tr_X.iloc[cutoff:]
  testY=tr_Y.iloc[cutoff:]
  model.fit(trainX, trainY)
  predY=model.predict(testX)
  mse=rmspe(predY, testY)
  print("MSE: %9f" % mse)
  #plotFeatureImportance(model,trainX,suffix)
  return model

def BuildMultiModel(tr,te,predict=False):
  #split the training set based on StoreType and then train each model separately.
  tr_stA=tr[tr['StoreType=a']==1]
  tr_stB=tr[tr['StoreType=b']==1]
  tr_stC=tr[tr['StoreType=c']==1]
  tr_stD=tr[tr['StoreType=d']==1]
  #OptimalBuildModel(tr_stA,'A')
  #OptimalBuildModel(tr_stB,'B')
  #OptimalBuildModel(tr_stC,'C')
  #OptimalBuildModel(tr_stD,'D')
  names =tr_stA.columns
  ''' 
  tr_stA=tr_stA.reindex(np.random.permutation(tr_stA.index))
  rlassoA = RandomizedLasso(alpha='bic',scaling=0.5,n_jobs=-1,sample_fraction=0.1)
  rlassoA.fit(tr_stA.drop('logSales',axis=1).iloc[:10000] , tr_stA['logSales'].iloc[:10000])
  print("Features sorted by their score A:")
  print(sorted(zip(map(lambda x: round(x, 4), rlassoA.scores_), names), reverse=True))
  
  names =tr_stB.columns
  tr_stB=tr_stB.reindex(np.random.permutation(tr_stB.index))
  rlassoB = RandomizedLasso(alpha='bic',scaling=0.5,n_jobs=-1,sample_fraction=0.1)
  rlassoB.fit(tr_stB.drop('logSales',axis=1).iloc[:10000] , tr_stB['logSales'].iloc[:10000])
  print("Features sorted by their score B:")
  print(sorted(zip(map(lambda x: round(x, 4), rlassoB.scores_), names), reverse=True))
  
  '''
  regA=GradientBoostingRegressor(n_estimators=10, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7,warm_start=True)
  regB=GradientBoostingRegressor(n_estimators=10, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7,warm_start=True)
  regC=GradientBoostingRegressor(n_estimators=10, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7,warm_start=True)
  regD=GradientBoostingRegressor(n_estimators=10, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7,warm_start=True)
  #figureBestParams(tr_stA,GradientBoostingRegressor(n_estimators=500,warm_start=True),'A')
  #figureBestParams(tr_stB,GradientBoostingRegressor(n_estimators=500,warm_start=True),'B')
  #train the respective models
  BuildModel(regA,tr_stA,'A')
  BuildModel(regB,tr_stB,'B')
  BuildModel(regC,tr_stC,'C')
  BuildModel(regD,tr_stD,'D')
  if predict==True:
    regA=GradientBoostingRegressor(n_estimators=10, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7,warm_start=True)
    regB=GradientBoostingRegressor(n_estimators=10, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7,warm_start=True)
    regC=GradientBoostingRegressor(n_estimators=10, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7,warm_start=True)
    regD=GradientBoostingRegressor(n_estimators=10, max_depth=9, max_features='auto',min_samples_split=7,min_samples_leaf=7,warm_start=True)
    a_Y=tr_stA['logSales']
    a_X=tr_stA.drop('logSales',axis=1)
    regA.fit(a_X, a_Y)
    b_Y=tr_stB['logSales']
    b_X=tr_stB.drop('logSales',axis=1)
    regB.fit(b_X, b_Y)
    c_Y=tr_stC['logSales']
    c_X=tr_stC.drop('logSales',axis=1)
    regC.fit(c_X, c_Y)
    d_Y=tr_stD['logSales']
    d_X=tr_stD.drop('logSales',axis=1)
    regD.fit(d_X, d_Y)
    te_stA=te[te['StoreType=a']==1]
    te_stB=te[te['StoreType=b']==1]
    te_stC=te[te['StoreType=c']==1]
    te_stD=te[te['StoreType=d']==1]
    pA=PredictTestSet(regA,te_stA,'A')
    pB=PredictTestSet(regB,te_stB,'B')
    pC=PredictTestSet(regC,te_stC,'C')
    pD=PredictTestSet(regD,te_stD,'D')
    inpX=pd.concat([te_stA, te_stB,te_stC,te_stD])
    predY=pd.concat([pA,pB,pC,pD])
    inpX['Id'].to_csv('inpxxx.csv')
    predY.to_csv('predyyyy.csv')

def compareModels(mdlA, df):
  # train and compare cmspe for the given models.
  df=df.reindex(np.random.permutation(df.index))
  tr_Y=df['logSales']
  tr_X=df.drop('logSales',axis=1)
  cutoff=tr_Y.size 
  cutoff=math.floor(0.8*cutoff)
  trainX=tr_X.iloc[0:cutoff]
  trainY=tr_Y.iloc[0:cutoff]
  testX=tr_X.iloc[cutoff:]
  testY=tr_Y.iloc[cutoff:]
  mdlA.fit(tr_X, tr_Y)
  mseA=rmspe(mdlA.predict(testX), testY)
  return mseA



def OptimalBuildModel(df,suffix='x'):
  #shuffle the dataset
  error_train=[]
  for k in range(100, 2001, 50):
    df=df.reindex(np.random.permutation(df.index))
    tr_Y=df['logSales']
    tr_X=df.drop('logSales',axis=1)
    print('Trying estimators ', k)
    kf=cross_validation.KFold(n=tr_Y.size,n_folds=5,shuffle=True) 
    err_cv=[]
    for tr_idx, te_idx in kf:
      stuff= str(tr_idx) + ' and ' + str(te_idx)
      print(stuff)
      trainX, testX=tr_X.ix[tr_idx], tr_X.ix[te_idx]
      trainY, testY=tr_Y[tr_idx], tr_Y[te_idx]
      clf = GradientBoostingRegressor( n_estimators=k,max_depth=2,max_features='sqrt',min_samples_leaf=7,min_samples_split=7)
      clf.fit(trainX, trainY)
      predY=clf.predict(testX)
      err=rmspe(predY, testY)
      err_cv.append(err)
      print(err)
    error_train.append(np.mean(err_cv))
  #Plot the data
  x=range(100,2001, 50)
  plt.hold(False)
  plt.style.use('ggplot')
  plt.plot(x, error_train, 'k')
  plt.xlabel('Number of Estimators', fontsize=18)
  plt.ylabel('Error', fontsize=18)
  plt.title('Error vs. Number of Estimators', fontsize=20)
  gfile='NumEstimatorsGraph'+suffix+'.png'
  plt.savefig(gfile)
  plt.cla()
  plt.clf()

def PredictTestSet(model,df,suffix='x'):
  xfile='InputX'+suffix+'.csv'
  yfile='predY'+suffix+'.csv'
  test=df.drop('Id',axis=1)
  predY=model.predict(test)
  t=[]
  for y in predY:
    t.append(math.exp(y))
  t2=pd.DataFrame(t)
  df.to_csv(xfile)
  t2.to_csv(yfile)
  return t2

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
  plt.show()

# Thanks to Chenglong Chen for providing this in the forum
def ToWeight(y):
  w = np.zeros(y.shape, dtype=float)
  ind = y != 0
  w[ind] = 1./(y[ind]**2)
  return w

def rmspe(yhat, y):
  w = ToWeight(y)
  rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
  return rmspe

def prepAllData():
	train=prepData('../orig/train.csv.zip', '../orig/store.csv.zip','train.csv')
	test=prepData('../orig/test.csv.zip', '../orig/store.csv.zip','test.csv')
	return train, test

def dataExploration(tr):
  #print(tr.head())
  #sns.pairplot(tr).savefig('pairplot.png')
  #sns.jointplot(x='logCompDist',y='logSales',data=tr).savefig('jointplot.png')
  sns.jointplot(x='StoreType',y='logSales',data=tr).savefig('storetype.png')
  

def figureBestParams(te, reg,suffix='x'):
  #lets do gridsearch to get the best parameters
  param_grid={ 'min_samples_leaf':[5,7,9,11],
				  'max_depth':[8,10,12,14,16],
				  'learning_rate': [0.1, 0.05,0.01,0.025]
			}
  tr_Y=te['logSales']
  tr_X=te.drop('logSales',axis=1)
  X_train, X_test, y_train, y_test = train_test_split(tr_X, tr_Y, test_size=0.5, random_state=0)
  print('Starting grid search ...'+suffix)
  clf = GridSearchCV(reg, param_grid, cv=5)
  #clf = RandomizedSearchCV(reg, param_grid, n_iter=4,n_jobs=8)
  clf.fit(X_train, y_train)
  print("Best parameters set found on development set:")
  print()
  print(clf.best_params_)
  print()
  print("Detailed classification report:")
  print()
  print("The model is trained on the full development set.")
  print("The scores are computed on the full evaluation set.")
  print()
  y_true, y_pred = y_test, clf.predict(X_test)
  mse=rmspe(y_pred, y_true)
  print("MSE: %6f" % mse)
  print()

def testMultiModels(tr):
  #split training data based on store type and compare randomforest and gradientboosting approaches.
  #plot performance for various estimators/load
  tr_stA=tr[tr['StoreType=b']==0]
  tr_stB=tr[tr['StoreType=b']==1]
  err=[]
  for i in range(100,351,50):
    mdlA_A=GradientBoostingRegressor(n_estimators=i, max_depth=9, max_features='auto',min_samples_split=5,min_samples_leaf=7,warm_start=True)
    err.append(compareModels(mdlA_A, tr_stA))
  x=range(100,351, 50)
  plt.hold(False)
  plt.style.use('ggplot')
  plt.plot(x, err, 'k')
  plt.xlabel('Number of Estimators', fontsize=18)
  plt.ylabel('Error', fontsize=18)
  plt.title('Error vs. Number of Estimators', fontsize=20)
  plt.savefig('Model A')
  plt.cla()
  plt.clf()

def calculateClusters(te):
  #lets try Birch clustering
  tr_Y=te['logSales']
  tr_X=te.drop('logSales',axis=1)
  sample=tr_X.iloc[:1000]
  print('create Birch')
  birch_model=Birch(threshold=1.7, n_clusters=None)
  print('Start fitting')
  birch_model.fit(sample)
  #plot the graph
  # Plot result
  labels = birch_model.labels_
  centroids = birch_model.subcluster_centers_
  n_clusters = np.unique(labels).size
  print("n_clusters : %d" % n_clusters)
  # Use all colors that matplotlib provides by default.
  colors_ = cycle(colors.cnames.keys())
  fig = plt.figure(figsize=(12, 4))
  fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)
  ax = fig.add_subplot(1, 3,  1)
  for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
    mask = labels == k
    ax.plot(sample[mask, 0], sample[mask, 1], 'w', markerfacecolor=col, marker='.')
    if birch_model.n_clusters is None:
      ax.plot(this_centroid[0], this_centroid[1], '+', markerfacecolor=col, markeredgecolor='k', markersize=5)
      ax.set_ylim([-25, 25])
      ax.set_xlim([-25, 25])
      ax.set_autoscaley_on(False)
      ax.set_title('Birch ')
  plt.savefig('BirchCluster.png')
  plt.clf()


def testGB():
  tr, te = prepAllData()
  #f, ax = plt.subplots(figsize=(14,14))
  #sns.heatmap(tr.corr(),vmax=0.8,square=True)
  #plt.savefig('RossmannCorrelation.png')
  #dataExploration(tr)
  #reg2=GradientBoostingRegressor(n_estimators=350, max_depth=9,max_features='auto',min_samples_split=7,min_samples_leaf=8,verbose=1)
  #reg2=BuildModel(reg2,tr)
  BuildMultiModel(tr,te,True)
  #OptimalBuildModel(tr)
  #reg3=GradientBoostingRegressor(n_estimators=350, max_depth=9,max_features='auto',min_samples_split=7,min_samples_leaf=8,verbose=1)
  #tr_Y=tr['logSales']
  #tr_X=tr.drop('logSales',axis=1)
  #reg3.fit(tr_X, tr_Y)
  #PredictTestSet(reg3,te)
  #figureBestParams(tr,GradientBoostingRegressor(n_estimators=400,warm_start=True))
  #testMultiModels(tr)
  #calculateClusters(tr)

tr=pd.DataFrame()
te=pd.DataFrame()

testGB()
