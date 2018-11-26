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

def birch(dataset,dataset_norm,num_clusters,caso):
    br = Birch(n_clusters = num_clusters)
    t = time.time()
    labels = br.fit(dataset_norm)
    tiempo = time.time() - t
    f = open("./resultados/"+caso+"/birch.txt","w")
    f.write("Ejecutando BIRCH")
    f.write(": {:.2f} segundos, ".format(tiempo))
    metric_CH = metrics.calinski_harabaz_score(dataset_norm, labels)
    f.write("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH)+"\n")
    metric_SC = metrics.silhouette_score(dataset_norm,cluster_predict,metric='euclidean',
        sample_size = floor(0.2*len(dataset_norm)), random_state = 77145416)
    f.write("Silhouette Coefficient: {:.5f}".format(metric_SC)+"\n")
    clusters = pd.DataFrame(labels,index = dataset.index, columns=['cluster'])
    f.write("Tamaño de cada cluster:\n")
    size = clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        f.write('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters))+"\n")
    f.close()





















#'''
