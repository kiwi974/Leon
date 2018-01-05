#La fonction de ce script permet d'effectuer la selection des variables

import data
import classement
import matplotlib.pyplot as plt
import numpy as np
import listeOperation as lo



def selectionVar(nbVarSonde):

    #Recuperation des spectres et du vecteur de sortie associe
    df = data.DataSet()
    Z = []
    print(df.loc[:,'H1'][1:])
    nbExemples = len(df.index)
    print(nbExemples)
    nbHarmoniques = df.shape[1]-1;
    for i in range(1,nbHarmoniques+1):
        exemple = []
        colonne = df.loc[:,'H'+str(i)]
        for j in range(1,nbExemples+1):
            exemple.append(float(colonne[j]))
        Z.append(exemple)
    sorties = df.loc[:,'Sortie']
    y = []
    for j in range(1,nbExemples):
        y.append(float(sorties[j]))

    #Calcul de la distribution des variables non pertinentes
    distrib = classement.distriNonPertinentes(y,Z,nbVarSonde)
    print(distrib)

    x = [i for i in range(nbHarmoniques)]
    proportions = lo.mult(distrib,[(1/nbVarSonde) for i in range(nbHarmoniques)])
    plt.plot(np.array(x),np.array(proportions))
    plt.show()

selectionVar(1000)

##y,Z = data.construcVA()
#classement.classer(y,Z,classement.genSonde(10))
