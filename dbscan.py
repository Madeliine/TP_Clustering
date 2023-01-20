# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:35:36 2023

@author: Madeline
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib . pyplot as plt
from scipy . io import arff
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors


path = './artificial/'
databrut = arff . loadarff ( open ( path + "chainlink.arff" , 'r') )
datanp = [ [ x[0] ,x[1]] for x in databrut [0] ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
datanp = np.asarray(datanp)
f0 = datanp [:,0] # tous les elements de la premiere colonne
f1 = datanp [:,1] # tous les elements de la deuxieme colonne
plt.scatter( f0, f1, s = 8 )
plt.title("Donnees initiales ")
plt.show()


#Distances k plus proches voisins
# Donnees dans X
k = 5
neigh = NearestNeighbors(n_neighbors = k)
neigh.fit(datanp)
distances , indices = neigh.kneighbors(datanp)
# retirer le point " origine "
newDistances = np.asarray ( [ np.average ( distances [ i ] [ 1 : ] ) for i in range (0, distances.shape[0])] )
trie = np.sort(newDistances)
plt.title(" Plus proches voisins ( 5 ) ")
plt.plot(trie) ;
plt.show()


silhouette_min = []
davies_bouldin_score_min = []
nb_clusters_min = []
X_min = np.arange(5, 15, 1)

silhouette_eps = []
davies_bouldin_score_eps = []
nb_clusters_eps = []
X_eps = np.arange(0.1, 1, 0.1)

#Variation de eps
for d in X_eps:

    clustering = DBSCAN(eps=d, min_samples=5).fit(datanp)
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    nb_clusters_eps.append(n_clusters_)
    silhouette_eps.append(metrics.silhouette_score(datanp, labels))
    davies_bouldin_score_eps.append(metrics.davies_bouldin_score(datanp, labels))

#Variations de min_sample
for d in X_min:

    clustering = DBSCAN(eps=0.09, min_samples=d).fit(datanp)
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    nb_clusters_min.append(n_clusters_)
    silhouette_min.append(metrics.silhouette_score(datanp, labels))
    davies_bouldin_score_min.append(metrics.davies_bouldin_score(datanp, labels))
    
    
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = labels == k
    
        xy = datanp[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )
    
        xy = datanp[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
    
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()



plt.bar(X_min, nb_clusters_min, width = 0.75)
plt.grid()
plt.xlabel('valeur de min_sample')
plt.ylabel('Nb de clusters')
plt.title ( "Nb de clusters en fonction de min_sample" )
plt.show ()

plt.bar(X_min, silhouette_min, width = 0.75)
plt.grid()
plt.xlabel('valeur de min_sample')
plt.ylabel('coef de Silhouette')
plt.title ( "Comparaison du coefficient de silhouette pour différentes valeurs de min_sample" )
plt.show ()

plt.bar(X_min, davies_bouldin_score_min, width = 0.75)
plt.grid()
plt.xlabel('valeur de min_sample')
plt.ylabel('Indice de Davies Bouldin')
plt.title ( "Comparaison de l'indice de Davies Bouldin pour différentes valeurs de min_sample " )
plt.show ()


plt.bar(X_eps, nb_clusters_eps, width = 0.075)
plt.grid()
plt.xlabel('valeur de eps')
plt.ylabel('Nb de clusters')
plt.title ( "Nb de clusters en fonction de eps" )
plt.show ()

plt.bar(X_eps, silhouette_eps, width = 0.075)
plt.grid()
plt.xlabel('valeur de eps')
plt.ylabel('coef de Silhouette')
plt.title ( "Comparaison du coefficient de silhouette pour différentes valeurs de eps" )
plt.show ()

plt.bar(X_eps, davies_bouldin_score_eps, width = 0.075)
plt.grid()
plt.xlabel('valeur de eps')
plt.ylabel('Indice de Davies Bouldin')
plt.title ( "Comparaison de l'indice de Davies Bouldin pour différentes valeurs de eps " )
plt.show ()