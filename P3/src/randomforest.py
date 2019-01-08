import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import preprocesamiento
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Preprocesado


X_train = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tra.csv')
X_test = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tst.csv')
y_train = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_tra_target.csv')
del y_train['id']


X_train, X_test = preprocesamiento.construction(X_train, X_test)
X_train, X_test = preprocesamiento.bools(X_train, X_test)
X_train, X_test = preprocesamiento.locs(X_train, X_test)
X_train, X_test = preprocesamiento.codes(X_train,X_test)
X_train['population'] = np.log(X_train['population'])
X_test['population'] = np.log(X_test['population'])
X_train, X_test = preprocesamiento.remov2(X_train, X_test)
#X_train, X_test = preprocesamiento.peques(X_train, X_test)
# cols = ['gps_height', 'latitude', 'longitude']
#X_train, X_test = preprocesamiento.pca(X_train, X_test, y_train)
X_train, X_test = preprocesamiento.lda(X_train, X_test, y_train)
X_train, X_test = preprocesamiento.dummies(X_train, X_test)

print(len(X_train.columns))

# Random Forest y ajuste de hiperparámetros
"""
rf = RandomForestClassifier(criterion='gini',
                                max_features='auto',
                                min_samples_split=6,
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

parametros = {"n_estimators" : [500, 750, 1000]}

grid = GridSearchCV(estimator=rf,
                  param_grid=parametros,
                  scoring='accuracy',
                  cv=2,
                  n_jobs=-1)

grid = grid.fit(X_train, y_train.values.ravel())


print(grid.best_score_)
print(grid.best_params_)
print(grid.cv_results_)
"""
# Modelo

modelo_random_forest = RandomForestClassifier(criterion='gini',
                                min_samples_split=6,
                                n_estimators=1000,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

modelo_random_forest.fit(X_train, y_train.values.ravel())
print("%.4f" % modelo_random_forest.oob_score_)

# Comprobando importancia de características

pd.concat((pd.DataFrame(X_train.columns, columns = ['variable']),
           pd.DataFrame(modelo_random_forest.feature_importances_, columns = ['importance'])),
          axis = 1).sort_values(by='importance', ascending = False)[:10]

# Submission file

prediccion = modelo_random_forest.predict(X_test)
y_test = pd.read_csv('/home/luisbalru/Universidad/Business-Intelligence/P3/data/water_pump_submissionformat.csv')
pred = pd.DataFrame(predictions, columns = [y_test.columns[1]])
del y_test['status_group']
y_test = pd.concat((y_test, pred), axis = 1)
y_test.to_csv(os.path.join('/home/luisbalru/Universidad/Business-Intelligence/P3/data/submission_files', 'submission4.csv'), sep=",", index = False)
