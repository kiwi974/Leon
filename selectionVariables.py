#La fonction de ce script permet d'effectuer la selection des variables

import data
import classement
import matplotlib.pyplot as plt
import numpy as np
import listeOperation as lo



def selectionVar(nbHarmoniques,nbVarSonde):

    #Recuperation des spectres et du vecteur de sortie associe
    y,Z = data.construcVA(nbHarmoniques)

    #Calcul de la distribution des variables non pertinentes
    distrib = classement.distriNonPertinentes(y,Z,nbVarSonde)
    print(distrib)

    x = [i for i in range(nbHarmoniques)]
    proportions = lo.mult(distrib,[(1/nbVarSonde) for i in range(nbHarmoniques)])
    plt.plot(np.array(x),np.array(proportions))
    plt.show()

selectionVar(30,1000)

##y,Z = data.construcVA()
#classement.classer(y,Z,classement.genSonde(10))
