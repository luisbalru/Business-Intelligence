# -*- coding_ utf-8 -*-
"""
Autor:
    Luis Balderas Ruiz
Fecha:
    Noviembre/2018
Contenido:
    Caso 3 de estudio: Hogar biparental con hijos menores de 25 años
    Práctica 2
    Inteligencia de Negocio
    Doble Grado en Ingeniería Informática y Matemáticas
    Universidad de Granada
"""

import pandas as pd
import numpy as np
import kmeans as km
import meanshift as ms
import birch as br
import dbscan as db
import jerarquico as j
import os

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

if(not(os.path.isdir("./resultados/Caso3"))):
    os.mkdir("./resultados/Caso3")

censo = pd.read_csv('./data/censo_granada.csv')
censo = censo.replace(np.NaN,0)

subset = censo.loc[censo['HIJOS'] <= 25]
subset = subset.loc[censo['ESTHOG'] == 8]

usadas = ['NPFAM', 'H6584', 'CMUNN', 'ANOCONS', 'ESTADO', 'DIFEDAD']
X = subset[usadas]
X_norm = X.apply(norm_to_zero_one)
caso = "Caso3"

# KMEANS
# Para calcular el número idóneo de clusters -> tras la ejecución se puede ver que es 2
print("kMeans")
km.kMeans(X,X_norm,caso)

# MEAN SHIFT
print("Mean Shift")
ms.meanshift(X,X_norm,caso)

# BIRCH
print("Birch")
br.birch(X,X_norm,4,caso)

# DBSCAN
print("DBSCAN")
db.dbscan(X,X_norm,0.2,50,caso)

# CLUSTERING JERÁRQUICO. WARD
print("Clustering jerárquico. Ward")
j.agglomerativeClustering(X,100,caso)
