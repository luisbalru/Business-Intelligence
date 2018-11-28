# -*- coding_ utf-8 -*-
"""
Autor:
    Luis Balderas Ruiz
Fecha:
    Noviembre/2018
Contenido:
    Clustering Jerárquico. AgglomerativeClustering con Ward como criterio de enlace
    Práctica 2
    Inteligencia de Negocio
    Doble Grado en Ingeniería Informática y Matemáticas
    Universidad de Granada
"""
import time

import pandas as pd
import os
import numpy as np

from sklearn import cluster
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy

def norma_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())



def agglomerativeClustering(X,num_clusters,caso):
    if(not(os.path.isdir("./resultados/"+caso+"/Jerarquico"))):
        os.mkdir("./resultados/"+caso+"/Jerarquico")
    if(len(X) > 1000):
        X = X.sample(1000, random_state=77145416)
    X_norm = X.apply(norma_to_zero_one)
    ward = cluster.AgglomerativeClustering(n_clusters = num_clusters, linkage='ward')
    name, algorithm = ('Ward', ward)
    cluster_predict = {}
    k = {}
    t = time.time()
    cluster_predict[name] = algorithm.fit_predict(X_norm)
    tiempo = time.time() - t
    k[name] = len(set(cluster_predict[name]))

    clusters = pd.DataFrame(cluster_predict['Ward'],index=X.index,columns=['cluster'])
    size = clusters['cluster'].value_counts()
    min_size = 10
    X_cluster = pd.concat([X_norm, clusters], axis=1)
    X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
    k_filtrado = len(set(X_filtrado['cluster']))

    X_filtrado = X_filtrado.drop('cluster', axis=1)
    X_filtrado_normal = X_filtrado.apply(norma_to_zero_one)
    X_filtrado_normal = X_filtrado_normal.replace(np.NaN,0)
    linkage_array = hierarchy.ward(X_filtrado_normal)
    plt.figure(1)
    plt.clf()
    h_dict = hierarchy.dendrogram(linkage_array,orientation='left')
    plt.show()

    n_cl = input("Dime el número de clusters que quieres: ")
    n_cl = int(n_cl)
    ward = cluster.AgglomerativeClustering(n_clusters = n_cl, linkage='ward')
    name, algorithm = ('Ward', ward)
    cluster_predict = {}
    k = {}
    t = time.time()
    cluster_predict[name] = algorithm.fit_predict(X_norm)
    tiempo = time.time() - t
    k[name] = len(set(cluster_predict[name]))
    f = open("./resultados/"+caso+"/Jerarquico/jerarquico-ward.txt","w")
    f.write("Ejecutando Clustering Aglomerativo")
    f.write(": k: {:3.0f}, \n".format(k[name]))
    f.write("{:6.2f} segundos\n".format(tiempo))
    clusters = pd.DataFrame(cluster_predict['Ward'],index=X.index,columns=['cluster'])
    size = clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        f.write('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters))+"\n")
    f.write("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k['Ward'],k_filtrado,min_size,len(X),len(X_filtrado)))
    f.close()
    X_cluster = pd.concat([X_norm, clusters], axis=1)
    cluster_centers = X_cluster.groupby('cluster').mean()
    centers = pd.DataFrame(cluster_centers,columns=list(X))
    centers_desnormal = centers.copy()


    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    ax = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    figure = ax.get_figure()
    figure.savefig("./resultados/"+caso+"/Jerarquico/heatmap-jerarquico.png", dpi=400)
    min_size = 50
    X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
    k_filtrado = len(set(X_filtrado['cluster']))
    sns_plot = sns.clustermap(X_filtrado_normal, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
    sns_plot.savefig("./resultados/"+caso+"/Jerarquico/clustermap.png")

#'''
