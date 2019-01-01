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

# Checking for Missing Data and Visualizing the Spread



NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
NAs[NAs.sum(axis=1) > 0]
display(NAs)

# Checking Initial correlations in Numerical Features

corr = train.select_dtypes(include = ['float64', 'int64']).iloc[:,1:].corr()
sns.set(font_scale=2)
sns.heatmap(corr,  linewidths=0.3, square=True)
plt.rcParams["figure.figsize"] = (30,20)
plt.show()

# Checking Distribution of Target Variable in Training Data

sns.countplot('status_group', data = training_labels)
plt.rcParams["figure.figsize"] = (30,30)
plt.show()

## The distribution is unequal and we will have to decide on classification metrics for comparing models

# LIMPIEZA DE DATOS

# Inspecting Cardinality of Categorical Data

cat_df = pd.DataFrame(columns=["Feature", "Cardinality","% Missings"])

total_cardinality = 0

i=0

for col in train.columns:
    if (train[col].dtype == np.object):
        cat_df.loc[i,"Feature"] = col
        cat_df.loc[i,"Cardinality"] = len(train[col].unique())
        total_cardinality += len(train[col].unique())
        pct_of_missing_values = float((len(train[col]) - train[col].count()) / len(train[col]))
        cat_df.loc[i,"% Missings"] = pct_of_missing_values*100
        i+=1

print("Total cardinality of categorical features:",total_cardinality)

display(cat_df)

#Combining Training and Testing Data for ease of imputations and feature engineering
combineddf=pd.concat([train,test],axis=0,ignore_index=True)
combineddf.head(5)

# Dropping categories with too few or too many factor levels

combineddf.drop(['wpt_name',    # too many levels
    'subvillage',  # too many levels; we have lat and long for location
    'ward',        # too many levels; we have lat and long for location
    'recorded_by', # constant
    'scheme_name', # too many levels
    'num_private', # irrelevant
    'region_code', # too many levels; we have lat and long for location
    'quantity_group', #same as quantity column
    'source_type',   #same as source but with fewer levels
    'waterpoint_type_group', #same as waterpoint
    'payment_type'],   #same as payment
              axis=1, inplace=True)

combineddf.shape

combinedtemp= combineddf['construction_year'].replace(0,nan)
combinedtemp1=combinedtemp.dropna(how='all',axis=0)
combinedtemp1.median()
#We will use this to replace 0's in construction year in the combined df

combinedtemp2= combineddf['gps_height'].replace(0,nan)
combinedtemp3=combinedtemp2.dropna(how='all',axis=0)
combinedtemp3.median()
#We will use this to replace 0's in gps height in the combined df

# Replacing Na's, Extracting Features and Converting Datatypes for better representation

#Replace Na's with 0
combineddf['funder'].replace(nan,0, inplace= True)
combineddf['installer'].replace(nan,0, inplace= True)
combineddf['permit'].replace(nan,0, inplace= True)
combineddf['scheme_management'].replace(nan,0, inplace= True)

#contains True or False values
combineddf['public_meeting'].fillna(value=True, inplace=True)

#Replace 0's with arbitrary categories
combineddf['permit'].replace("0","TRUE", inplace= True) #corresponding to 0's almost entirely not pay to use - so let's consider as True
combineddf['public_meeting'].replace("0","TRUE", inplace= True)
combineddf['scheme_management'].replace(0, "Unknown", inplace=True)  #large number of Blanks - cannot fit this into None or Other group
combineddf['funder'].replace(0, "Other", inplace=True)
combineddf['installer'].replace(0, "Other", inplace=True)
combineddf['installer'].replace('-', "Other", inplace=True)
combineddf['construction_year'].replace(0, 2000, inplace=True) #median value for values aside from 0
combineddf['gps_height'].replace(0, 1166, inplace=True)        #median value for values aside from 0

combineddf['date_recorded'] = pd.to_datetime(combineddf['date_recorded'])   #converting date_recorded into number of days since first recorded date
combineddf['date_recorded'] = (combineddf['date_recorded'] - combineddf['date_recorded'].min()) / np.timedelta64(1, 'D')

#Type-conversion
combineddf['population']=combineddf['population'].astype('float64')  #we all float down here
combineddf['gps_height']=combineddf['gps_height'].astype('float64')
combineddf['construction_year']=combineddf['construction_year'].astype('float64')
combineddf['amount_tsh']=combineddf['amount_tsh'].astype('float64')

display(combineddf)




combineddf.groupby(['extraction_type_class','extraction_type_group'])['extraction_type'].value_counts()

#Define function to replace repeating values under common levels within extracting type - make extraction type group obsolete
def clean_values(combineddf, col, values_dict):
    for k, v in values_dict.items():
        combineddf.loc[combineddf[col] == k, col] = v

#Define key-value pairs for replacing
clean_values(combineddf, 'extraction_type',
    {
        'india mark ii'             : 'india',
        'india mark iii'            : 'india',
        'other - swn 81'            : 'swn',
        'swn 80'                    : 'swn',
        'other - play pump'         : 'other handpump',
        'walimi'                    : 'other handpump',
        'other - mkulima/shinyanga' : 'other handpump',
        'cemo'                      : 'other motorpump',
        'climax'                    : 'other motorpump',
    }
)


#Drop 'extraction_type_group'
combineddf.drop(['extraction_type_group'],axis=1, inplace=True)
combineddf.shape

combineddf.groupby(['extraction_type_class'])['extraction_type'].value_counts()

combineddf.isnull().sum()

#Latitude and Longitude to cartesian coordinates
# Assuming Earth as sphere not ellipsoid
def cartesian_x(lat,lon):
    lat=radians(lat)
    lon=radians(lon)
    R=6371.0
    x = R * cos(lat) * cos(lon)
    return x
def cartesian_y(lat,lon):
    lat=radians(lat)
    lon=radians(lon)
    R=6371.0
    y = R * cos(lat) * sin(lon)
    return y

# extracting cartesian x,y cordinates form latitude and longitude
combineddf['x1']=[cartesian_x(i,j) for i,j in zip(combineddf['latitude'],combineddf['longitude'])]
combineddf['y1']=[cartesian_y(i,j) for i,j in zip(combineddf['latitude'],combineddf['longitude'])]

#Manhattan distance as a new feature
#combineddf['Manhattan_dist'] =(combineddf['x1']).abs() +(combineddf['y1']).abs()


combineddf.drop(['latitude',
    'longitude'
    #,'x1',
    #'y1'
                ], axis=1, inplace=True)

display(combineddf['funder'].describe())


#Listing Categorical columns for label encoding
cat_cols=combineddf.select_dtypes(include=['object']).columns.values.tolist()
cat_cols

# Treating Categorical and Numerical Features

#Listing numerical columns for scaling
num_cols=combineddf.select_dtypes(include=['int64', 'float64']).columns.values.tolist()
num_cols

#Listing Categorical columns for label encoding
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
 'installer')  #need to encode funder and installer to pass through models

# Applying Label Encoding
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(combineddf[c].values))
    combineddf[c] = lbl.transform(list(combineddf[c].values))

###########################
#funder and installer had too many levels; going back to format them as function of frequency
###########################
combineddf.shape

combineddf.dtypes

# Scaling numeric features
scaler = StandardScaler()
numeric_features = combineddf.select_dtypes(include=np.float64)

scaler.fit(numeric_features)
combineddf[numeric_features.columns] = scaler.transform(combineddf[numeric_features.columns])

display(combineddf[numeric_features.columns].head())



#Encoding status_group of train to pass it through our models
y = training_labels.status_group
LEncoder = LabelEncoder()
LEncoder.fit(y)
labels = LEncoder.transform(y)
print(LEncoder.classes_)



display(labels)



# Preparing Training, Validation and Test Data for Modeling


x = combineddf[:train.shape[0]]
test_data = combineddf[train.shape[0]:]


x.to_csv("pythonoutputtrain.csv")
test_data.to_csv("pythonoutputtest.csv")

temp_forid = combineddf[train.shape[0]:]
test_ids= temp_forid.id

# Split training and validation data
x, labels = shuffle(x, labels, random_state= 10)
X_train, X_valid, y_train, y_valid = train_test_split(x, labels, train_size=0.8, random_state=10)

## BUILDING MODELS



RF_model = RandomForestClassifier(n_estimators=1000)
RF_model.fit(X_train, y_train)
y_pred = RF_model.predict(X_valid)

#Accuracy on validation data
acc=accuracy_score(y_valid, y_pred)
fscore = fbeta_score(y_valid, y_pred, average='weighted', beta=0.5)
print('RF Accuracy:', acc)
print('RF Fbeta score:', fscore)

#this is my base model - score was ~80.1 with this on the leaderboard.Not going with this for final submission unless I do an ensemble; used it to infer feature importances
#manhattan distance has lower importance than latitude longitude (based on past trials)
#but it's represented better, so will keep manhattan distance as is; plus CatBoost performs better with this feature than lat and long

#Finally decided to remove Manhattan distance and just keep the cartesian coordinates, because catboost did not perform well



features = RF_model.feature_importances_
imp_features = np.argsort(features)[::-1]
k = 20
print(x[:train.shape[0]].columns[imp_features[:10]])
series_imp_features = pd.Series(features[imp_features[:k]], x[:train.shape[0]].columns[imp_features[:k]])
series_imp_features.plot(kind='bar')

# catboost model

cbc= CatBoostClassifier(
    learning_rate=0.1,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    od_pval=0.01,
    random_seed=10
)



cbc.fit(X_train, y_train)

#Validation score with CatBoost
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
       'funder', 'installer'] #adding funder and installer as categories again; representing them as function of frequencies did not help with catboost



ctrs_indexes = []
for i, v in enumerate(features):
    if v in ctrs:
        ctrs_indexes.append(i)

#fitting whole train set
cbc = CatBoostClassifier(
    learning_rate=0.1,
    loss_function='MultiClass',
    eval_metric='Accuracy',
).fit(x, labels, cat_features= ctrs_indexes)   #got over-fitted model when features was not specified; possibly because it considers everything as a category by default

def submit(pred, name='ans_catboost'):
    y_pred = LEncoder.inverse_transform(pred.astype(int))
    ans_catboost = pd.DataFrame({'id': test_ids, 'status_group': y_pred.ravel()})
    ans_catboost.to_csv('submissions/' + name + '.csv', index=False)

submit(cbc.predict(test_data))

#tried hyperparameter optimization; catboost does not perform well for multiclass, especially with mixed categorical and numerical types
#level of tuning required outweighed by performance and ease of use of other base models
#even RandomForest performed better in this case, so moving onto XGBoost

## XGBOOST

param = {'booster': 'gbtree',
        'obective': 'multi:softmax',
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

## BAYESIAN-OPTIMIZATION FOR hyperparameter

#Defining function for optimizing XGBoost - can also do this for other models like RF
def xgbcv(max_depth,
          learning_rate,
          n_estimators,
          min_child_weight,
          gamma,
          subsample,
          colsample_bytree,
          reg_alpha,
          reg_lambda,
          silent=True):
    return cv_s(xgb.XGBClassifier(max_depth=int(max_depth),
                                 learning_rate=learning_rate,
                                 n_estimators=int(n_estimators),
                                 gamma=gamma,
                                 reg_alpha=reg_alpha,
                                 min_child_weight=min_child_weight,
                                 objective='multi:softmax'),
                    x,
                    labels,
                    "accuracy",
                    cv=4).mean()

##Setting range of parameters for optimization
xgboostBO = BayesOpt(xgbcv,
                                 {
                                  'max_depth': (4,23),
                                  'learning_rate': (0.01, 0.2),
                                  'n_estimators': (200,600),
                                  'gamma': (0.01, 10),
                                  'min_child_weight': (1,40),
                                  'subsample': (0.2, 1),
                                  'colsample_bytree' :(0.2, 1),
                                  'reg_alpha':(0, 10),
                                  'reg_lambda':(0, 10)
                                  })

print ("Start Optimization of Main Model")
#xgboostBO.maximize(init_points=10,n_iter=110, xi=0.0,  acq="poi")

#Best parameters from Bayes-opt for accuracy as a metric
param = {
                  'max_depth' : 19,
                  'learning_rate': 0.0486,
                  #'n_estimators': 591,  #removed and mentioned as num_boost_round; same thing
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

importances = clf_final.get_fscore()
importances = sorted(importances.items(), key=operator.itemgetter(1))

temp = pd.DataFrame(importances, columns=['feature', 'fscore'])
temp['fscore'] = temp['fscore'] / temp['fscore'].sum()
plt.figure()
temp.plot()
temp.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
