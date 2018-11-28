# -*- coding_ utf-8 -*-
"""
Autor:
    Luis Balderas Ruiz
Fecha:
    Noviembre/2018
Contenido:
    Clustering con el algoritmo MeanShift
    Práctica 2
    Inteligencia de Negocio
    Doble Grado en Ingeniería Informática y Matemáticas
    Universidad de Granada
"""
import time
import numpy as np
import pandas as pd
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from math import floor
from itertools import cycle
from sklearn.cluster import MeanShift, estimate_bandwidth
import seaborn as sns


def meanshift(dataset,dataset_norm,caso):
    if(not(os.path.isdir("./resultados/"+caso+"/MeanShift"))):
        os.mkdir("./resultados/"+caso+"/MeanShift")
    bandwidth = estimate_bandwidth(dataset_norm, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
    t = time.time()
    cluster_predict = ms.fit(dataset_norm)
    tiempo = time.time() - t
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    f = open("./resultados/"+caso+"/MeanShift/meanshift.txt","w")
    f.write("Ejecutando MeanShift")
    f.write(": {:.2f} segundos, ".format(tiempo)+"\n")
    f.write("Número estimado de clusters: %d \n" %n_clusters_)
    metric_CH = metrics.calinski_harabaz_score(dataset_norm, labels)
    f.write("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH)+"\n")
    metric_SC = metrics.silhouette_score(dataset_norm,labels,metric='euclidean',
        sample_size = floor(0.2*len(dataset_norm)), random_state = 77145416)
    f.write("Silhouette Coefficient: {:.5f}".format(metric_SC)+"\n")
    clusters = pd.DataFrame(labels,index = dataset.index, columns=['cluster'])
    f.write("Tamaño de cada cluster:\n")
    size = clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        f.write('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters))+"\n")
    f.close()

    centers = pd.DataFrame(cluster_centers,columns=list(dataset))
    centers_desnormal = centers.copy()


    for var in list(centers):
        centers_desnormal[var] = dataset[var].min() + centers[var] * (dataset[var].max() - dataset[var].min())

    ax = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    figure = ax.get_figure()
    figure.savefig("./resultados/"+caso+"/MeanShift/heatmap-ms.png", dpi=400)

    X_meanshift = pd.concat([dataset, clusters], axis=1)
    sns.set()
    variables = list(X_meanshift)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_meanshift, vars=variables, hue="cluster") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("./resultados/"+caso+"/MeanShift/meanshift.png")


#'''
