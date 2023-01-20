# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:36:30 2023

@author: Madeline
"""

import numpy as np
import matplotlib . pyplot as plt
from scipy . io import arff
import time
from sklearn import cluster
from sklearn import metrics
import scipy . cluster . hierarchy as shc
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
path = './artificial/'
databrut = arff . loadarff ( open ( path + "twodiamonds.arff" , 'r') )
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


# Donnees dans datanp
print ( " Dendrogramme ’ single ’ donnees initiales " )
linked_mat = shc.linkage ( datanp , 'single')
plt.figure (figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat , orientation = 'top',distance_sort = 'single' , show_leaf_counts = False )
plt.show ()

silhouette_dist = []
davies_bouldin_score_dist = []
silhouette_cl = []
davies_bouldin_score_cl = []

X_dist = np.arange(0.1, 0.9, 0.1)
X_cluster = np.arange(2, 7, 1)

# set di stance_threshold ( 0 ensures we compute the full tree )
nb_cluster = []
for d in X_dist:
    tps1 = time.time ()
    model = cluster.AgglomerativeClustering( distance_threshold = d , linkage = 'single' , n_clusters = None )
    model = model.fit ( datanp )
    tps2 = time.time ()
    labels = model.labels_
    k = model.n_clusters_
    leaves = model.n_leaves_
    
    silhouette_dist.append(metrics.silhouette_score(datanp, labels))
    davies_bouldin_score_dist.append(metrics.davies_bouldin_score(datanp, labels))
    nb_cluster.append(k)
    
    plt.scatter ( f0 , f1 , c = labels , s = 8 )
    plt.title ( " Resultat du clustering " )
    plt.show ()
    print ( "nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
    

#set the number of clusters
for k in X_cluster:
    tps1 = time.time ()
    model = cluster.AgglomerativeClustering( linkage = 'single' , n_clusters = k )
    model = model.fit ( datanp )
    tps2 = time.time ()
    labels = model.labels_
    kres = model.n_clusters_
    leaves = model.n_leaves_
    
    silhouette_cl.append(metrics.silhouette_score(datanp, labels))
    davies_bouldin_score_cl.append(metrics.davies_bouldin_score(datanp, labels))
    


# Affichage clustering    
    plt.scatter ( f0 , f1 , c = labels , s = 8 )
    plt.title ( " Resultat du clustering " )
    plt.show ()
    print ( "nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
    #runtime_tot.append(round (( tps2 - tps1 ) * 1000 , 2 ))
    
plt.bar(X_cluster, silhouette_cl, width = 0.75)
plt.grid()
plt.xlabel('Nombre de clusters')
plt.ylabel('coef de Silhouette')
plt.title ( "Comparaison du coefficient de silhouette pour différentes valeurs de k (nombre de clusters) " )
plt.show ()

plt.bar(X_cluster, davies_bouldin_score_cl, width = 0.75)
plt.grid()
plt.xlabel('Nombre de clusters')
plt.ylabel('Indice de Davies Bouldin')
plt.title ( "Comparaison de l'indice de Davies Bouldin pour différentes valeurs de k (nombre de clusters) " )
plt.show ()

plt.bar(X_dist, silhouette_dist, width = 0.075)
plt.grid()
plt.xlabel('Seuil de distance')
plt.ylabel('coef de Silhouette')
plt.title ( "Comparaison du coefficient de silhouette pour différentes valeurs de d (distance threshold) " )
plt.show ()

plt.bar(X_dist, davies_bouldin_score_dist, width = 0.075)
plt.grid()
plt.xlabel('Seuil de distance')
plt.ylabel('Indice de Davies Bouldin')
plt.title ( "Comparaison de l'indice de Davies Bouldin pour différentes valeurs de d (distance threshold) " )
plt.show ()

plt.bar(X_dist, nb_cluster, width = 0.075)
plt.grid()
plt.xlabel('Seuil de distance')
plt.ylabel('Nombre de clusters')
plt.title ( "Nombres de clusters pour différentes valeurs de d (distance threshold) " )
plt.show ()
