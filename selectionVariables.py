#La fonction de ce script permet d'effectuer la selection des variables

import data
import classement
import matplotlib.pyplot as plt
import numpy as np
import listeOperation as lo
import sqlite3
import os


#Changement du repertoire de travaille
os.chdir("/home/ray974/Learning/Data")


def recupererData(chemin,nbParts):
    conn = sqlite3.connect(chemin)
    cursor = conn.cursor()

    Z = [[] for i in range(nbParts+1)]
    y = []

    cursor.execute("""SELECT moyenne_freq_ponderee, densites FROM male""")
    for row in cursor:
        ex = row[1].split("<->")
        for j in range(nbParts):
            Z[j].append(float(ex[j]))
        Z[nbParts].append(float(row[0]))
        y.append(1)

    cursor.execute("""SELECT moyenne_freq_ponderee, densites FROM female""")
    for row in cursor:
        ex = row[1].split("<->")
        for j in range(nbParts):
            Z[j].append(float(ex[j]))
        Z[nbParts].append(float(row[0]))
        y.append(-1)

    return y,Z


#recupererData("bdd_dev.db",10)

def selectionVar(chemin,nbParts,nbVarSonde):

    #Recuperation des spectres et du vecteur de sortie associe
    y,Z = recupererData(chemin,nbParts)

    #Calcul de la distribution des variables non pertinentes
    print("Debut du classement...")
    distrib = classement.distriNonPertinentes(y,Z,nbVarSonde)
    print("La distribution trouvee est : " + str(distrib))

    x = [i for i in range(nbParts+2)]
    proportions = lo.mult(distrib,[(1/nbVarSonde) for i in range(nbParts+2)])
    plt.plot(np.array(x),np.array(proportions))
    plt.show()



selectionVar("bdd_dev.db",10,4)

##y,Z = data.construcVA()
#classement.classer(y,Z,classement.genSonde(10))
