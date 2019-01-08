import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA


def remov2(X_train, X_test):
    z = ['id','amount_tsh',  'num_private', 'region',
          'quantity', 'quality_group', 'source_type', 'payment',
          'waterpoint_type_group',
         'extraction_type_group']
    for i in z:
        del X_train[i]
        del X_test[i]
    return X_train, X_test


def construction(X_train, X_test):
    for i in [X_train, X_test]:
        i['construction_year'].replace(0, X_train[X_train['construction_year'] != 0]['construction_year'].mean(), inplace=True)
    return X_train, X_test


def locs(X_train, X_test):
    trans = ['longitude', 'latitude', 'gps_height', 'population']
    for i in [X_train, X_test]:
        i.loc[i.longitude == 0, 'latitude'] = 0
    for z in trans:
        for i in [X_train, X_test]:
            i[z].replace(0., np.NaN, inplace = True)
            i[z].replace(1., np.NaN, inplace = True)

        for j in ['subvillage', 'district_code', 'basin']:

            X_train['mean'] = X_train.groupby([j])[z].transform('mean')
            X_train[z] = X_train[z].fillna(X_train['mean'])
            o = X_train.groupby([j])[z].mean()
            fill = pd.merge(X_test, pd.DataFrame(o), left_on=[j], right_index=True, how='left').iloc[:,-1]
            X_test[z] = X_test[z].fillna(fill)

        X_train[z] = X_train[z].fillna(X_train[z].mean())
        X_test[z] = X_test[z].fillna(X_train[z].mean())
        del X_train['mean']
    return X_train, X_test

def bools(X_train, X_test):
    z = ['public_meeting', 'permit']
    for i in z:
        X_train[i].fillna(False, inplace = True)
        X_train[i] = X_train[i].apply(lambda x: float(x))
        X_test[i].fillna(False, inplace = True)
        X_test[i] = X_test[i].apply(lambda x: float(x))
    return X_train, X_test


def codes(X_train, X_test):
    for i in ['region_code', 'district_code']:
        X_train[i] = X_train[i].apply(lambda x: str(x))
        X_test[i] = X_test[i].apply(lambda x: str(x))
    return X_train, X_test

def dummies(X_train, X_test):
    columns = [i for i in X_train.columns if type(X_train[i].iloc[0]) == str]
    for column in columns:
        X_train[column].fillna('NULL', inplace = True)
        good_cols = [column+'_'+i for i in X_train[column].unique() if i in X_test[column].unique()]
        X_train = pd.concat((X_train, pd.get_dummies(X_train[column], prefix = column)[good_cols]), axis = 1)
        X_test = pd.concat((X_test, pd.get_dummies(X_test[column], prefix = column)[good_cols]), axis = 1)
        del X_train[column]
        del X_test[column]
    return X_train, X_test

def lda(X_train, X_test, y_train, cols=['population', 'gps_height', 'latitude', 'longitude']):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train[cols])
    X_test_std = sc.transform(X_test[cols])
    lda = LDA(n_components=None)
    X_train_lda = lda.fit_transform(X_train_std, y_train.values.ravel())
    X_test_lda = lda.transform(X_test_std)
    X_train = pd.concat((pd.DataFrame(X_train_lda), X_train), axis=1)
    X_test = pd.concat((pd.DataFrame(X_test_lda), X_test), axis=1)
    for i in cols:
        del X_train[i]
        del X_test[i]
    return X_train, X_test

def pca(X_train,X_test,y_train, cols=['population','gps_height','latitude','longitude']):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train[cols])
    X_test_std = sc.transform(X_test[cols])
    pca = PCA(n_components=None)
    X_train_lda = pca.fit_transform(X_train_std, y_train.values.ravel())
    X_test_lda = pca.transform(X_test_std)
    X_train = pd.concat((pd.DataFrame(X_train_lda), X_train), axis=1)
    X_test = pd.concat((pd.DataFrame(X_test_lda), X_test), axis=1)
    for i in cols:
        del X_train[i]
        del X_test[i]
    return X_train, X_test


