#La fonction de ce script permet d'effectuer la selection des variables

import data
import classement
import matplotlib.pyplot as plt
import numpy as np
import listeOperation as lo



def selectionVar(chemin,nbVarSonde):

    #Recuperation des spectres et du vecteur de sortie associe
    print("Extraction des données...")
    df = data.DataSet(chemin)
    nbExemples = len(df.index)
    nbHarmoniques = df.shape[1]-1;

    Z = []
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
    print("Debut du classement...")
    distrib = classement.distriNonPertinentes(y,Z,nbVarSonde)
    print("La distribution trouvee est : " + str(distrib))

    x = [i for i in range(nbHarmoniques)]
    proportions = lo.mult(distrib,[(1/nbVarSonde) for i in range(nbHarmoniques)])
    plt.plot(np.array(x),np.array(proportions))
    plt.show()

selectionVar("Data/data",1000)

##y,Z = data.construcVA()
#classement.classer(y,Z,classement.genSonde(10))
