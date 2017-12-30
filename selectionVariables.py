#La fonction de ce script permet d'effectuer la selection des variables

import data
import classement
import matplotlib.pyplot as plt
import numpy as np



def selectionVar():

    #Recuperation des spectres et du vecteur de sortie associe
    y,Z = data.construcVA()

    #Calcul de la distribution des variables non pertinentes
    distrib = classement.distriNonPertinentes(y,Z,1000)
    print(distrib)

    x = [i for i in range(30)]
    plt.plot(np.array(x),np.array(distrib))
    plt.show()

selectionVar()

##y,Z = data.construcVA()
#classement.classer(y,Z,classement.genSonde(10))
