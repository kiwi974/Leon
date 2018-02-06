#La fonction de ce script permet d'effectuer la selection des variables

import data
import classement
import matplotlib.pyplot as plt
import numpy as np
import listeOperation as lo
import exploitation as explo
import os


#Changement du repertoire de travaille
os.chdir("/home/ray974/Learning/Data")


def selectionVar(chemin,nbVarSonde):

    #Recuperation des spectres et du vecteur de sortie associe
    y,Z,nbDesc = explo.getDataVar(chemin)

    #Calcul de la distribution des variables non pertinentes
    print("Debut du classement...")
    distrib, classements = classement.distriNonPertinentes(y,Z,nbVarSonde)
    print("La distribution trouvee est : " + str(distrib))

    plt.figure(figsize=(9, 7))

    nbParts = nbDesc

    #Traces variable sonde
    x = [(i+1) for i in range(nbParts+1)]
    proportions = lo.mult(distrib,[(1/nbVarSonde) for i in range(nbParts+2)])
    plt.plot(np.array(x),np.array(proportions),'deeppink')

    #Traces pour les autres variables
    couleurs = ['black','rosybrown','brown','red','saddlebrown','darkorange','orange','goldenrod','olive','greenyellow',
                'yellow','forestgreen','turquoise','cyan','royalblue','darkblue','darkorchid','darkviolet','purple',
                'pink']
    rangs = [[0 for j in range(nbParts+1)] for i in range(nbParts+1)]
    for i in range(len(classements)):
        for j in range(len(classements[i])):
            rangs[classements[i][j]-1][j] += 1
    for i in range(len(rangs)):
        rangs[i] = lo.mult(rangs[i],[(1/nbVarSonde) for i in range(nbParts+1)])
    for i in range(nbParts+1):
        if (i<=14):
            lab = 'd'+str(i+1)
        else :
            lab = 'moy'
        plt.plot(np.array(x),np.array(rangs[i]),couleurs[i],label=lab)

    plt.legend()

    plt.show()



#selectionVar("bdd_dev.db",20)

