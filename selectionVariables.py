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
    cursor.execute("""SELECT moyenne_freq_ponderee, densites FROM male""")
    Z = [[] for i in range(nbParts+1)]
    for row in cursor:
        ex = row[1].split("<->")
        for j in range(nbParts):
            Z[j].append(ex[j])
        Z[nbParts].append(row[0])
    return Z


recupererData("bdd_dev.db",10)

def selectionVar(chemin,nbVarSonde):

    #Recuperation des spectres et du vecteur de sortie associe
    y = []
    Z = []

    #Calcul de la distribution des variables non pertinentes
    print("Debut du classement...")
    distrib = classement.distriNonPertinentes(y,Z,nbVarSonde)
    print("La distribution trouvee est : " + str(distrib))

    x = [i for i in range(4)]
    proportions = lo.mult(distrib,[(1/nbVarSonde) for i in range(4)])
    plt.plot(np.array(x),np.array(proportions))
    plt.show()

#selectionVar("Data/data",1000)

##y,Z = data.construcVA()
#classement.classer(y,Z,classement.genSonde(10))
