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
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.utils import shuffle


train = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tra.csv')
test = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tst.csv')
training_labels = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tra_target.csv')

train.shape
