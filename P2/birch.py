# -*- coding_ utf-8 -*-
"""
Autor:
    Luis Balderas Ruiz
Fecha:
    Noviembre/2018
Contenido:
    Clustering con el algoritmo BIRCH
    Práctica 2
    Inteligencia de Negocio
    Doble Grado en Ingeniería Informática y Matemáticas
    Universidad de Granada
"""
import time
import numpy as np
import pandas as pd

from sklearn import metrics
import matplotlib.pyplot as plt
from math import floor
from itertools import cycle
from sklearn.cluster import Birch
import seaborn as sns


def birch(dataset,dataset_norm,num_clusters,caso):
    if(not(os.path.isdir("./resultados/"+caso+"/BIRCH"))):
        os.mkdir("./resultados/"+caso+"/BIRCH")
    br = Birch(n_clusters = num_clusters)
    t = time.time()
    labels = br.fit_predict(dataset_norm)
    tiempo = time.time() - t
    f = open("./resultados/"+caso+"/BIRCH/birch.txt","w")
    f.write("Ejecutando BIRCH")
    f.write(": {:.2f} segundos, ".format(tiempo))
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
    X_birch = pd.concat([dataset_norm,clusters],axis=1)
    edad = X_birch.groupby(['cluster'])['EDAD'].mean()
    esreal = X_birch.groupby(['cluster'])['ESREAL'].mean()
    cmunn = X_birch.groupby(['cluster'])['CMUNN'].mean()
    estumad = X_birch.groupby(['cluster'])['ESTUMAD'].mean()
    estupad = X_birch.groupby(['cluster'])['ESTUPAD'].mean()
    cluster_centers = pd.concat([edad,esreal,cmunn,estumad,estupad],axis=1)

    centers = pd.DataFrame(cluster_centers,columns=list(dataset))
    centers_desnormal = centers.copy()


    for var in list(centers):
        centers_desnormal[var] = dataset[var].min() + centers[var] * (dataset[var].max() - dataset[var].min())

    ax = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    figure = ax.get_figure()
    figure.savefig("./resultados/"+caso+"/BIRCH/heatmap-br.png", dpi=400)

    X_birch = pd.concat([dataset,clusters],axis=1)
    sns.set()
    variables = list(X_birch)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_birch, vars = variables, hue="cluster")
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("./resultados/"+caso+"/BIRCH/birch.png")




















#'''
