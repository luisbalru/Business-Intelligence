# -*- coding_ utf-8 -*-
"""
Autor:
    Luis Balderas Ruiz
Fecha:
    Noviembre/2018
Contenido:
    Caso 1 de estudio
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
import os

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

if(not(os.path.isdir("./resultados/Caso1"))):
    os.mkdir("./resultados/Caso1")

censo = pd.read_csv('./data/censo_granada.csv')
censo = censo.replace(np.NaN,0)

subset = censo.loc[censo['EDADMAD'] > 0]
subset = subset.loc[censo['EDADPAD'] > 0]

usadas = ['EDAD', 'ESREAL', 'CMUNN', 'ESTUMAD', 'ESTUPAD']
X = subset[usadas]
X_norm = X.apply(norm_to_zero_one)

# KMEANS
# Para calcular el número idóneo de clusters -> tras la ejecución se puede ver que es 2
 # km.ElbowMethod(X)
caso = "Caso1"
#km.kMeans(X,X_norm,2,caso)

# MEAN SHIFT
#ms.meanshift(X,X_norm,caso)

# BIRCH
br.birch(X,X_norm,4,caso)
