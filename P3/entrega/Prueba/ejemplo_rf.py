import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

le = preprocessing.LabelEncoder()

'''
lectura de datos
'''
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('water_pump_tra.csv')
data_y = pd.read_csv('water_pump_tra_target.csv')
data_x_tst = pd.read_csv('water_pump_tst.csv')

#se quitan las columnas que no se usan
data_x.drop(labels=['id'], axis=1,inplace = True)
data_x.drop(labels=['date_recorded'], axis=1,inplace = True)

data_x_tst.drop(labels=['id'], axis=1,inplace = True)
data_x_tst.drop(labels=['date_recorded'], axis=1,inplace = True)

data_y.drop(labels=['id'], axis=1,inplace = True)
    
'''
Se convierten las variables categóricas a variables numéricas (ordinales)
'''
from sklearn.preprocessing import LabelEncoder
mask = data_x.isnull()
data_x_tmp = data_x.fillna(9999)
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform)
data_x_nan = data_x_tmp.where(~mask, data_x)

mask = data_x_tst.isnull() #máscara para luego recuperar los NaN
data_x_tmp = data_x_tst.fillna(9999) #LabelEncoder no funciona con NaN, se asigna un valor no usado
data_x_tmp = data_x_tmp.astype(str).apply(LabelEncoder().fit_transform) #se convierten categóricas en numéricas
data_x_tst_nan = data_x_tmp.where(~mask, data_x_tst) #se recuperan los NaN

X = data_x_nan.values
X_tst = data_x_tst_nan.values
y = np.ravel(data_y.values)

#------------------------------------------------------------------------
'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
le = preprocessing.LabelEncoder()

def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("Score: {:.4f}, tiempo: {:6.2f} segundos".format(accuracy_score(y[test],y_pred) , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all
#------------------------------------------------------------------------

imp = Imputer(missing_values='NaN', strategy='most_frequent')
imp = imp.fit(X)
X_train_imp = imp.transform(X)
imp = imp.fit(X_tst)
X_test_imp = imp.transform(X_tst)

clf = RandomForestClassifier(n_estimators=200,max_depth=5,random_state=0)
clf = clf.fit(X_train_imp,y)
y_pred_tra = clf.predict(X_train_imp)
print("Score: {:.4f}".format(accuracy_score(y,y_pred_tra)))
y_pred_tst = clf.predict(X_test_imp)

df_submission = pd.read_csv('water_pump_submissionformat.csv')
df_submission['status_group'] = y_pred_tst
df_submission.to_csv("submission2.csv", index=False)

