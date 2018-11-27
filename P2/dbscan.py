# -*- coding_ utf-8 -*-
"""
Autor:
    Luis Balderas Ruiz
Fecha:
    Noviembre/2018
Contenido:
    Clustering con el algoritmo DBSCAN
    Práctica 2
    Inteligencia de Negocio
    Doble Grado en Ingeniería Informática y Matemáticas
    Universidad de Granada
"""

import time
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from math import floor
from itertools import cycle
import seaborn as sns

def dbscan(dataset,dataset_norm,eps,min_samples,caso):
    if(not(os.path.isdir("./resultados/"+caso+"/DBSCAN"))):
        os.mkdir("./resultados/"+caso+"/DBSCAN")
    t = time.time()
    db = DBSCAN(eps=eps,min_samples=min_samples).fit(dataset_norm)
    labels = db.labels_
    tiempo = time.time() - t

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    f = open("./resultados/"+caso+"/DBSCAN/dbscan.txt","w")
    f.write("Ejecutando DBSCAN")
    f.write(": {:.2f} segundos, ".format(tiempo))
    metric_CH = metrics.calinski_harabaz_score(dataset_norm, labels)
    f.write("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH)+"\n")
    metric_SC = metrics.silhouette_score(dataset_norm,labels,metric='euclidean',
        sample_size = floor(0.2*len(dataset_norm)), random_state = 77145416)
    f.write("Silhouette Coefficient: {:.5f}".format(metric_SC)+"\n")
    clusters = pd.DataFrame(labels,index = dataset.index, columns=['cluster'])
    f.write("Epsilon: %f \n" %eps)
    f.write("Min_samples: %d \n"%min_samples)
    f.write("Número estimado de clusters: %d\n" %n_clusters_)
    f.write("Número estimado de puntos ruidosos: %d \n" %n_noise_)
    f.write("Tamaño de cada cluster:\n")
    size = clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        f.write('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters))+"\n")
    f.close()

    X_DBSCAN = pd.concat([dataset_norm,clusters],axis=1)
    X_DBSCAN = X_DBSCAN[X_DBSCAN.cluster != -1]
    edad = X_DBSCAN.groupby(['cluster'])['EDAD'].mean()
    esreal = X_DBSCAN.groupby(['cluster'])['ESREAL'].mean()
    cmunn = X_DBSCAN.groupby(['cluster'])['CMUNN'].mean()
    estumad = X_DBSCAN.groupby(['cluster'])['ESTUMAD'].mean()
    estupad = X_DBSCAN.groupby(['cluster'])['ESTUPAD'].mean()
    cluster_centers = pd.concat([edad,esreal,cmunn,estumad,estupad],axis=1)

    centers = pd.DataFrame(cluster_centers,columns=list(dataset))
    centers_desnormal = centers.copy()


    for var in list(centers):
        centers_desnormal[var] = dataset[var].min() + centers[var] * (dataset[var].max() - dataset[var].min())

    ax = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    figure = ax.get_figure()
    figure.savefig("./resultados/"+caso+"/DBSCAN/heatmap-dbscan.png", dpi=400)

    X_DBSCAN = pd.concat([dataset,clusters],axis=1)
    X_DBSCAN = X_DBSCAN[X_DBSCAN.cluster != -1]
    sns.set()
    variables = list(X_DBSCAN)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_DBSCAN, vars = variables, hue="cluster")
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("./resultados/"+caso+"/DBSCAN/dbscan.png")





















#'''
