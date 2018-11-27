# -*- coding_ utf-8 -*-
"""
Autor:
    Luis Balderas Ruiz
Fecha:
    Noviembre/2018
Contenido:
    Clustering con el algoritmo K-Means
    Práctica 2
    Inteligencia de Negocio
    Doble Grado en Ingeniería Informática y Matemáticas
    Universidad de Granada
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from math import floor
import seaborn as sns

def ElbowMethod(dataset_norm):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(dataset_norm)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def kMeans(dataset,dataset_norm,caso):
    if(not(os.path.isdir("./resultados/"+caso+"/kMeans"))):
        os.mkdir("./resultados/"+caso+"/kMeans")
    ElbowMethod(dataset_norm)
    n_cl = print("Indica el número de clusters: ")
    n_cl = int(n_cl)
    k_means = KMeans(init='k-means++', n_clusters=n_cl)
    t = time.time()
    cluster_predict = k_means.fit_predict(dataset_norm)
    tiempo = time.time() - t
    f = open("./resultados/"+caso+"/kMeans/kmeans.txt","w")
    f.write("Ejecutando k-means")
    f.write(": {:.2f} segundos, ".format(tiempo))
    metric_CH = metrics.calinski_harabaz_score(dataset_norm, cluster_predict)
    f.write("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH)+"\n")
    metric_SC = metrics.silhouette_score(dataset_norm,cluster_predict,metric='euclidean',
        sample_size = floor(0.2*len(dataset_norm)), random_state = 77145416)
    f.write("Silhouette Coefficient: {:.5f}".format(metric_SC)+"\n")
    clusters = pd.DataFrame(cluster_predict,index = dataset.index, columns=['cluster'])
    f.write("Tamaño de cada cluster:\n")
    size = clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        f.write('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters))+"\n")
    f.close()

    # HEATMAP CON CENTROIDES
    centers = pd.DataFrame(k_means.cluster_centers_,columns=list(dataset))
    centers_desnormal = centers.copy()

    for var in list(centers):
        centers_desnormal[var] = dataset[var].min() + centers[var] * (dataset[var].max() - dataset[var].min())

    ax = sns.heatmap(centers, cmap="YlGnBu", annot = centers_desnormal, fmt='.3f')
    figure = ax.get_figure()
    figure.savefig("./resultados/"+caso+"/kMeans/heatmap-km.png", dpi=400)

    # Scatter Plot
    X_kmeans = pd.concat([dataset,clusters],axis=1)
    sns.set()
    variables = list(X_kmeans)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_kmeans, vars = variables, hue = 'cluster')
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
    sns_plot.savefig("./resultados/"+caso+"/kMeans/kmeans.png")

#'''
