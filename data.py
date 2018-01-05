#Package dont les fonctions serviront au traitement des data qui seront utilisee
#pour l'apprentissage


import FFT
import math
import listeOperation as lo
import pandas as pd
import numpy as np
import os

#Changement du repertoire de travaille
os.chdir("/home/ray974/Learning/")

"""Fonction qui initialise le fichier data avec le nombre d'harmonqiues souhaitées, le nombre de données déjà 
présentes (0) et la liste des colonnes du DataFrame
param : nbHarmonqiues -> nombre d'harmoniques souhaitées pour la fft
!!! Les donnees deja presentes dans le fichier seront ecrasees!!!"""

def initializeData(nbHarmoniques):

    #Ouverture du fichier data
    data = open("Data/data",'w')

    #Au depart il n'y a aucune donnee
    data.write(str(0)+'\n')

    #On selectionnera nbHarmoniques harmoniques avec la fft
    data.write(str(nbHarmoniques)+'\n')

    #On signale autant de colonnes qu'il y a d'harmoniques souhaitees
    colonnes = ""
    for i in range(nbHarmoniques):
        if (len(colonnes)==0):
            colonnes = "H"+str(i+1)
        else:
            colonnes = colonnes + "<->" + "H" + str(i+1)

    #A chaque exemple sera associé une sortie que l'on doit preciser
    colonnes = colonnes + "<->" + "Sortie"

    #Il faut alors écrire ceci dans le fichier data
    data.write(colonnes+'\n')





"""Fonction qui effectue le prétraitement d'un donnee z : normalisation et reduction
param : z -> variable a traiter"""

def pretraitement(z):
    n = len(z)
    moy = 0
    for i in range(len(z)):
        moy = moy + z[i]
    moy = moy/n
    sigma = 0
    for i in range(len(z)):
        sigma = sigma + (z[i] - moy)**2
    sigma = math.sqrt(sigma/(n-1))
    res = lo.sous(z,[moy for i in range(n)])
    res = lo.div(res,[sigma for i in range(n)])
    return res





"""Fonction qui contruit les exemples d'apprentissage ainsi que le vecteur de sortie
y (similaire à constructVA mais construit les exemples, non les variables"""

def updateDataSet():

    #Acquisition des noms des fichiers
    Hfiles = open("VoiceRecord/homme/Hfiles",'r')
    Ffiles = open("VoiceRecord/femme/Ffiles",'r')

    #Fichiers dans lesquels on écrit les fichiers déhà traités
    Htreated = open("VoiceRecord/homme/Htreated",'a')
    Ftreated = open("VoiceRecord/femme/Ftreated",'a')

    #Ouverture du fichier destination pour y ajouter les resultats de la fft
    data = open("Data/data",'r')

    #Nombre de donnees deja presentes dans le fichier data
    nbData  = int(data.readline())

    #Nombre d'harmoniques selectionnee pour chaque jeu de donnees
    nbHarmoniques = int(data.readline())

    data.close()

    #On souhaite compter le nombre de fichier ajoutes
    compteurH = 0
    compteurF = 0

    data = open("Data/data",'a')

    #Acquisition des donnees concernant les hommes
    print("Hommes")
    nomH = Hfiles.readline()
    nomH = nomH[:len(nomH)-1]
    while (len(nomH) != 0):
        #Ecritude du nom dans le fichier des extraits deja traites
        Htreated.write(nomH+"\n")
        spectre = pretraitement(FFT.fftFreq("VoiceRecord/homme/"+nomH+".wav",nbHarmoniques))
        ligne = str(1)
        for i in range(nbHarmoniques):
            ligne = ligne + "<->" + str(spectre[i])
        ligne = ligne + "\n"
        compteurH += 1
        nbData += 1
        data.write(ligne)
        nomH = Hfiles.readline()
        nomH = nomH[:len(nomH)-1]
    print("On a traité " + str(compteurH) + " fichiers.")

    #Acquisition des donnees concernant les femmes
    print("Femmes")
    nomF = Ffiles.readline()
    nomF = nomF[:len(nomF)-1]
    while (len(nomF) != 0):
        #Ecritude du nom dans le fichier des extraits deja traites
        Ftreated.write(nomF+"\n")
        spectre = pretraitement(FFT.fftFreq("VoiceRecord/femme/"+nomF+".wav",nbHarmoniques))
        ligne = str(-1)
        for i in range(nbHarmoniques):
            ligne = ligne + "<->" + str(spectre[i])
        ligne = ligne + "\n"
        compteurF += 1
        nbData += 1
        data.write(ligne)
        nomF = Ffiles.readline()
        nomF = nomF[:len(nomF)-1]
    print("On a traité " + str(compteurF) + " fichiers.")

    #Fermeture des fichiers
    Hfiles.close()
    Ffiles.close()
    Htreated.close()
    Ftreated.close()
    data.close()

    #Effacement des fichiers qui contenaient les nouveaux enregistrements
    Hfiles = open("VoiceRecord/homme/Hfiles",'w')
    Ffiles = open("VoiceRecord/femme/Ffiles",'w')
    Hfiles.close()
    Ffiles.close()

    return 0

#updateDataSet()




























"""Fonction qui construit les realisations des differentes variables à partir des
spectres frequentiels"""

def construcVA(nbHarmoniques):

    #Vecteur des differentes realisations des variables
    Z = [[] for i in range(30)]

    #Vecteur des mesures : 1 pour les hommes et -1 pour les femmes
    y = []

    #Acquisition des donnees concernant les hommes
    print("Hommes")
    for k in range(len(fichierH)):
        spectre = FFT.fftFreq("/home/ray974/Learning/VoiceRecord/homme/"+fichierH[k]+".wav",nbHarmoniques)
        print(spectre)
        for i in range(len(spectre)):
            Z[i].append(spectre[i])
        y.append(1)

    print("Femmes")
    #Acquisition des donnees concernant les femmes
    for k in range(len(fichierF)):
        spectre = FFT.fftFreq("/home/ray974/Learning/VoiceRecord/femme/"+fichierF[k]+".wav",nbHarmoniques)
        print(spectre)
        for i in range(len(spectre)):
            Z[i].append(spectre[i])
        y.append(-1)

    #Pretraitement des donnees
    print("Pretraitement")
    for k in range(len(Z)):
        Z[k] = pretraitement(Z[k])

    print(Z)

    return y,Z


#y,Z = construcVA()
#print(np.array(y))
#print(np.array(Z))

