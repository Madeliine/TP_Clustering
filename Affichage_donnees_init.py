# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 08:07:15 2023

@author: Madeline
"""

import numpy as np
import matplotlib . pyplot as plt
from scipy . io import arff

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
databrut = arff . loadarff ( open ( path + "xclara.arff" , 'r') )
datanp = [ [ x[0] ,x[1]] for x in databrut [0] ]
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
datanp = np.asarray(datanp)
f0 = datanp [:,0] # tous les elements de la premiere colonne
f1 = datanp [:,1] # tous les elements de la deuxieme colonne
plt.scatter( f0, f1, s = 8)
plt.title("Donnees initiales ")
plt.xlabel("f1")
plt.ylabel("f2")
plt.show()