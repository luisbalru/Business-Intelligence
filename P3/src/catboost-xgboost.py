import pandas as pd
import missingno as msno
import numpy as np
from numpy import nan
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from math import radians,cos,sin
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from IPython.display import display
from scipy.stats import boxcox
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
import warnings  #ignore all the warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, make_scorer, fbeta_score
from catboost import Pool, CatBoostClassifier, cv
from itertools import combinations
import xgboost as xgb
from sklearn.cross_validation import cross_val_score as cv_s
from bayes_opt import BayesianOptimization as BayesOpt
import operator
from sklearn.utils import shuffle

# EXPLORACIÓN INICIAL DE LOS DATOS

train = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tra.csv')
test = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tst.csv')
training_labels = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tra_target.csv')

train.shape
test.shape

# Comprobando valores perdidos



NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
NAs[NAs.sum(axis=1) > 0]
display(NAs)

# Correlaciones iniciales entre variables numéricas

corr = train.select_dtypes(include = ['float64', 'int64']).iloc[:,1:].corr()
sns.set(font_scale=2)
sns.heatmap(corr,  linewidths=0.3, square=True)
plt.rcParams["figure.figsize"] = (30,20)
plt.show()

# Comprobando la distribución de la variable de clasificación

sns.countplot('status_group', data = training_labels)
plt.rcParams["figure.figsize"] = (30,30)
plt.show()


# LIMPIEZA DE DATOS

# Estudiando cardinalidad de las variables categóricas

cat_df = pd.DataFrame(columns=["Característica", "Cardinal","MV"])

total_cardinality = 0

i=0

for col in train.columns:
    if (train[col].dtype == np.object):
        cat_df.loc[i,"Característica"] = col
        cat_df.loc[i,"Cardinal"] = len(train[col].unique())
        total_cardinality += len(train[col].unique())
        pct_of_missing_values = float((len(train[col]) - train[col].count()) / len(train[col]))
        cat_df.loc[i,"MV"] = pct_of_missing_values*100
        i+=1

print("Variables categóricas:",total_cardinality)

display(cat_df)

#Combinando training y test
combineddf=pd.concat([train,test],axis=0,ignore_index=True)
combineddf.head(5)

# Variables a eliminar 

combineddf.drop(['wpt_name',    # demasiados niveles
    'subvillage',  # demasiados niveles
    'ward',        # demasiados niveles
    'recorded_by', # constante
    'scheme_name', # demasiados niveles
    'num_private', # irrelevante
    'region_code', # demasiados niveles
    'quantity_group', # redundante
    'source_type',   # redundante
    'waterpoint_type_group', #redundante
    'payment_type'],   #redundante
              axis=1, inplace=True)

combineddf.shape

combinedtemp= combineddf['construction_year'].replace(0,nan)
combinedtemp1=combinedtemp.dropna(how='all',axis=0)
combinedtemp1.median()


combinedtemp2= combineddf['gps_height'].replace(0,nan)
combinedtemp3=combinedtemp2.dropna(how='all',axis=0)
combinedtemp3.median()


# Cambiando NAN por 0
combineddf['funder'].replace(nan,0, inplace= True)
combineddf['installer'].replace(nan,0, inplace= True)
combineddf['permit'].replace(nan,0, inplace= True)
combineddf['scheme_management'].replace(nan,0, inplace= True)

combineddf['public_meeting'].fillna(value=True, inplace=True)


combineddf['permit'].replace("0","TRUE", inplace= True) 
combineddf['public_meeting'].replace("0","TRUE", inplace= True)
combineddf['scheme_management'].replace(0, "Unknown", inplace=True) 
combineddf['funder'].replace(0, "Other", inplace=True)
combineddf['installer'].replace(0, "Other", inplace=True)
combineddf['installer'].replace('-', "Other", inplace=True)
combineddf['construction_year'].replace(0, 2000, inplace=True)
combineddf['gps_height'].replace(0, 1166, inplace=True)       

combineddf['date_recorded'] = pd.to_datetime(combineddf['date_recorded'])   
combineddf['date_recorded'] = (combineddf['date_recorded'] - combineddf['date_recorded'].min()) / np.timedelta64(1, 'D')

# Conversión de tipos
combineddf['population']=combineddf['population'].astype('float64') 
combineddf['gps_height']=combineddf['gps_height'].astype('float64')
combineddf['construction_year']=combineddf['construction_year'].astype('float64')
combineddf['amount_tsh']=combineddf['amount_tsh'].astype('float64')

display(combineddf)




combineddf.groupby(['extraction_type_class','extraction_type_group'])['extraction_type'].value_counts()


#Elimino extraction_type_group
combineddf.drop(['extraction_type_group'],axis=1, inplace=True)
combineddf.shape

combineddf.groupby(['extraction_type_class'])['extraction_type'].value_counts()

combineddf.isnull().sum()


cat_cols=combineddf.select_dtypes(include=['object']).columns.values.tolist()
cat_cols

# Variables categóricas y numéricas

num_cols=combineddf.select_dtypes(include=['int64', 'float64']).columns.values.tolist()
num_cols


cols=('basin',
 'extraction_type',
 'extraction_type_class',
 'lga',
 'management',
 'management_group',
 'payment',
 'permit',
 'quality_group',
 'quantity',
 'region',
 'scheme_management',
 'source',
 'source_class',
 'water_quality',
 'waterpoint_type',
 'public_meeting',
 'funder',
 'installer')  

# Codificando variables
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(combineddf[c].values))
    combineddf[c] = lbl.transform(list(combineddf[c].values))


combineddf.shape

combineddf.dtypes

# Escalado de características
scaler = StandardScaler()
numeric_features = combineddf.select_dtypes(include=np.float64)

scaler.fit(numeric_features)
combineddf[numeric_features.columns] = scaler.transform(combineddf[numeric_features.columns])

display(combineddf[numeric_features.columns].head())





x = combineddf[:train.shape[0]]
test_data = combineddf[train.shape[0]:]


x, labels = shuffle(x, labels, random_state= 10)
X_train, X_valid, y_train, y_valid = train_test_split(x, labels, train_size=0.8, random_state=10)

# Modelo CatBoost

cbc= CatBoostClassifier(
    learning_rate=0.1,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    od_pval=0.01,
    random_seed=10
)



cbc.fit(X_train, y_train)

# Score
print('Train: {:.4f}'.format(
    cbc.score(X_train, y_train)
))



x.columns.values

ctrs= ['basin', 'construction_year',
       'district_code', 'extraction_type', 'extraction_type_class',
        'lga', 'management', 'management_group', 'payment',
       'permit', 'public_meeting', 'quality_group',
       'quantity', 'region', 'scheme_management', 'source',
       'source_class', 'water_quality', 'waterpoint_type',
       'funder', 'installer']



ctrs_indexes = []
for i, v in enumerate(features):
    if v in ctrs:
        ctrs_indexes.append(i)


cbc = CatBoostClassifier(
    learning_rate=0.1,
    loss_function='MultiClass',
    eval_metric='Accuracy',
).fit(x, labels, cat_features= ctrs_indexes)  

def submit(pred, name='ans_catboost'):
    y_pred = LEncoder.inverse_transform(pred.astype(int))
    ans_catboost = pd.DataFrame({'id': test_ids, 'status_group': y_pred.ravel()})
    ans_catboost.to_csv('submissions/' + name + '.csv', index=False)

submit(cbc.predict(test_data))


## XGBOOST

param = {'booster': 'gbtree',
        'objective': 'multi:softmax',
        'eta': 0.025,
        'max_depth': 23,
        'colsample_bytree': 0.4,
        'silent': 1,
        'eval_metric': 'mlogloss',
        'num_class': 3
        }

train_dmatrix = xgb.DMatrix(x, label=labels, missing=np.nan)

clf = xgb.train(param, dtrain=train_dmatrix, num_boost_round=400, maximize=False)

test_dmatrix = xgb.DMatrix(test_data, missing=np.nan)

def submit(pred, name='ans_xgb'):
    y_pred = LEncoder.inverse_transform(pred.astype(int))
    ans_xgb = pd.DataFrame({'id': test_ids, 'status_group': y_pred.ravel()})
    ans_xgb.to_csv('submissions/' + name + '.csv', index=False)

submit(clf.predict(test_dmatrix))

importances = clf.get_fscore()
importances = sorted(importances.items(), key=operator.itemgetter(1))



temp = pd.DataFrame(importances, columns=['feature', 'fscore'])
temp['fscore'] = temp['fscore'] / temp['fscore'].sum()
plt.figure()
temp.plot()
temp.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')

param = {
                  'max_depth' : 19,
                  'learning_rate': 0.0486,
                  'gamma': 0.0354,
                  'min_child_weight': 19.2455,
                  'subsample': 0.9842,
                  'colsample_bytree': 0.6716,
                  'reg_alpha': 6.0458,
                  'objective':'multi:softmax',
                  'num_class': 3,
                  'eval_metric': 'mlogloss'
        }

clf_final = xgb.train(param, dtrain=train_dmatrix, num_boost_round=591, maximize=False)

def submit(pred, name='ans_xgb_final'):
    y_pred = LEncoder.inverse_transform(pred.astype(int))
    ans_xgb_final = pd.DataFrame({'id': test_ids, 'status_group': y_pred.ravel()})
    ans_xgb_final.to_csv('submissions/' + name + '.csv', index=False)

submit(clf.predict(test_dmatrix))
"""
importances = clf_final.get_fscore()
importances = sorted(importances.items(), key=operator.itemgetter(1))

temp = pd.DataFrame(importances, columns=['feature', 'fscore'])
temp['fscore'] = temp['fscore'] / temp['fscore'].sum()
plt.figure()
temp.plot()
temp.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
"""
