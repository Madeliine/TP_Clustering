

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:25:03 2023

@author: delhay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time
from sklearn import cluster
from sklearn import metrics


# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information

path = './dataset-rapport/'
name_file = 'zz2.txt'
databrut = np.loadtxt(path + name_file)
datanp = [ [ x[0] ,x[1]] for x in databrut]
datanp = np.asarray(datanp)
f0 = datanp [:,0] # tous les elements de la premiere colonne
f1 = datanp [:,1] # tous les elements de la deuxieme colonne

#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension
# f1 : valeur sur la deuxieme dimension
#

print ( " Appel KMeans pour une valeur fixee de k (=nombre de clusters)" )

silhouette = []
davies_bouldin_score = []
calinski_harabasz_score = []
temps_calcul = []
nb_iteration = []

for k in range(2, 10):
    tps1 = time.time ()
    model = cluster.KMeans (n_clusters =k , init = 'k-means++')
    model.fit ( datanp )
    tps2 = time.time ()
    labels = model.labels_
    iteration = model.n_iter_
    plt.scatter ( f0 , f1 , c = labels , s = 8 )
    plt.title ( 'Donnees apres clustering Kmeans pour k = %i' %k)
    plt.show ()
    print ( " nb clusters = " ,k , " , nb iter = " , iteration , " ,runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
    silhouette.append(metrics.silhouette_score(datanp, labels))
    davies_bouldin_score.append(metrics.davies_bouldin_score(datanp, labels))
    temps_calcul.append(round (( tps2 - tps1 ) * 1000)) #en ms
    nb_iteration.append(iteration)
    
                        
    

    
    
X = range(2, 10)  
plt.bar(X, silhouette)
plt.xticks(X, X)
plt.title ( "Comparaison du coefficient de silhouette pour différents nombres de clusters " )
plt.xlabel("Nombre de clusters")
plt.ylabel("Valeur du coefficient de silhouette")
plt.grid()
plt.show ()

plt.bar(X, davies_bouldin_score)
plt.xticks(X, X)
plt.title ( "Comparaison de l'indice de Davies Bouldin pour différents nombres de clusters " )
plt.xlabel("Nombre de clusters")
plt.ylabel("Valeur de l'indice de Davies Bouldin")
plt.grid()
plt.show ()

plt.bar(X, temps_calcul)
plt.xticks(X, X)
plt.title ( "Comparaison du temps de calcul pour différents nombres de clusters" )
plt.xlabel("Nombre de clusters")
plt.ylabel("Temps de calcul en ms")
plt.grid()
plt.show ()

plt.bar(X, nb_iteration)
plt.xticks(X, X)
plt.title ( "Comparaison du nombre d'itération pour différents nombres de clusters" )
plt.xlabel("Nombre de clusters")
plt.ylabel("Nombre d'itération")
plt.grid()
plt.show ()
    
silhouette = np.asarray(silhouette)
davies_bouldin_score = np.asarray(davies_bouldin_score)
print("Le nombre de clusters optimal, d'après le coefficient de silhouette est", X[np.nanargmax(silhouette)])
print("Le nombre de clusters optimal, d'après l'indice de Davies Bouldin est", X[np.nanargmin(davies_bouldin_score)])
