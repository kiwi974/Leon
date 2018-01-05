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





""" Fonction qui construit l'ensemble des datas pour l'apprentissage
param : nbHarmoniques -> paramètres optionnel qui, s'il est spécifié, permet de ne garder que les nbHarmoniques 
                         premières harmoniques pour chaque exemple. Sinon, on le prend toutes.
precondition : nbHarmoniques <= nb total d'harmoniques presentes dans le jeu de donnees de base
                   --> on peut noter que ceci implique de toujours enlever le '\n' en fin de ligne si jamais il y a 
                       egalite dans la relation precedente '
Remarque : on pourra obtenir les variables aléatoires associées pour la selection des variables en selectionnant 
directement les colonnes (et la dernière colonne)"""

def DataSet(nbHarmoniques = -1):

    #Ouverture du fichier data
    data = open("Data/data",'r')

    #Recuperation du nb de data, du nb d'harmoniques et de la liste des colonnes
    nbData = int(data.readline())
    nbHarmo = int(data.readline())
    colonnes = data.readline().split("<->")
    n = len(colonnes[len(colonnes)-1])
    colonnes[len(colonnes)-1] = colonnes[len(colonnes)-1][:n-1]    #On enlève le '\n' qui est sur le dernier mot

    if (nbHarmoniques != -1):
        s = colonnes[len(colonnes)-1]
        colonnes = colonnes[:nbHarmoniques]
        colonnes.append(s)

    #indexage du dataFrame
    index = 1

    #Creation du dataFrame vide
    df = pd.DataFrame(data=[], index= [], columns=colonnes)

    #Construction des lignes du DataFrame
    exemple = data.readline()
    while (len(exemple) != 0):
        freq = exemple.split("<->")
        freq[len(freq)-1] = freq[len(freq)-1][:len(freq[len(freq)-1])-1]    #On enlève le '\n' qui est sur le dernier mot

        #On reduit le nombre d'harmoniques si besoin
        if (nbHarmoniques != -1):
            freq = freq[:nbHarmoniques+1]

        #On met la sortie à la fin du vecteur
        s = freq[0]
        freq.append(s)
        freq = freq[1:]

        #Ajout de l'exemple au DataFrame
        df.loc[index] = freq

        #Exemple suivant
        index += 1
        exemple = data.readline()
    return df



    #df.lock[index] = [frequences]


#DataSet(10)