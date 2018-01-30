import sqlite3
import os
import csv
import FFT
import listeOperation as lo
import exploitation as explo
import numpy as np

# Changement du repertoire de travaille
os.chdir("/home/ray974/Learning/Data")


def initaliazeBDD(name):
    bdd = sqlite3.connect(name)

    cursor = bdd.cursor()

    # Creation d'un table homme
    cursor.execute("""
          CREATE TABLE IF NOT EXISTS male(
          id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
          nom_extrait TEXT,
          frequences TEXT, 
          amplitudes TEXT,
          moyenne_freq_ponderee REAL,
          densites TEXT
          )
    """)

    # Creation d'un table femme
    cursor.execute("""
          CREATE TABLE IF NOT EXISTS female(
          id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
          nom_extrait TEXT,
          frequences TEXT, 
          amplitudes TEXT,
          moyenne_freq_ponderee REAL,
          densites TEXT
          )
    """)
    bdd.commit()

    bdd.close()


initaliazeBDD("bdd_dev.db")

""" Fonction ramenant les valeurs du tableau tab de l'intervalle [0,n] dans l'intervalle [0,1]. """


def toIntervalle(tab, n):
    tab = lo.div(tab, [n for i in range(len(tab))])
    # On ne garde que 5 chiffres après la virgule
    for i in range(len(tab)):
        tab[i] = ((tab[i] * 100000) // 1) / 100000
    return tab


def traitementEnregistrements(name, freqMin, freqMax, nbParts):
    bdd = sqlite3.connect(name)
    cursor = bdd.cursor()

    nbFiles = len(os.listdir("/media/ray974/common-voice/cv-valid-dev/wav"))

    mPondH = []
    mPondF = []

    descripteursH = []
    descripteursF = []

    nbH = 0
    nbF = 0

    with open("/media/ray974/common-voice/cv-valid-dev.csv", 'rt') as f:
        reader = csv.reader(f)
        nbRow = 0
        for col in reader:
            if (nbRow > nbFiles - 1):
                break
            ind = ""
            for i in range(6 - len(str(nbRow))):
                if (ind == ""):
                    ind = "0"
                else:
                    ind = ind + "0"
            if (col[5] == "male"):
                freq, spectre = FFT.fftFreq(
                    "/media/ray974/common-voice/cv-valid-dev/wav/sample-" + ind + str(nbRow) + ".wav", freqMin, freqMax)
                n = len(freq)
                mPondH.append(explo.pondMoy(freq, spectre, n))
                l = explo.densite(spectre, n, nbParts)
                if (len(l) != (nbParts-1)):
                    break
                descripteursH.append(l)
                freq = lo.tabToString(freq)
                spectre = lo.tabToString(spectre)
                chaine = "sample-" + ind + str(nbRow)
                cursor.execute("INSERT INTO male(nom_extrait, frequences, amplitudes) VALUES(?,?,?)", (chaine, freq, spectre))
                nbH += 1
            elif (col[5] == "female"):
                freq, spectre = FFT.fftFreq(
                    "/media/ray974/common-voice/cv-valid-dev/wav/sample-" + ind + str(nbRow) + ".wav", freqMin, freqMax)
                n = len(freq)
                mPondF.append(explo.pondMoy(freq, spectre, n))
                l = explo.densite(spectre, n, nbParts)
                if (len(l) != (nbParts-1)):
                    break
                descripteursF.append(l)
                freq = lo.tabToString(freq)
                spectre = lo.tabToString(spectre)
                chaine = "sample-" + ind + str(nbRow)
                cursor.execute("INSERT INTO female(nom_extrait, frequences, amplitudes) VALUES(?,?,?)", (chaine, freq, spectre))
                nbF += 1
            else:
                useless = 1
            nbRow += 1

    # Pretraitement des donnees
    n = len(descripteursH[0])  # = len(descripteursF[0])

    mPondF = explo.pretraitement(mPondF)
    mPondH = explo.pretraitement(mPondH)

    for j in range(n):

        #traitement de descripteurH
        u = []
        for i in range(len(descripteursH)):
            u.append(descripteursH[i][j])
        u = explo.pretraitement(u)
        for i in range(len(descripteursH)):
            descripteursH[i][j] = u[i]

        #traitement de descripteurF
        v = []
        for i in range(len(descripteursF)):
            v.append(descripteursF[i][j])
        v = explo.pretraitement(v)
        for i in range(len(descripteursF)):
            descripteursF[i][j] = v[i]

    for i in range(1, nbH + 1):
        cursor.execute("""UPDATE male SET moyenne_freq_ponderee = ? WHERE id = ?""", (mPondH[i - 1], i))
        cursor.execute("""UPDATE male SET densites = ? WHERE id = ?""", (lo.tabToString(descripteursH[i - 1]), i))
    for i in range(1, nbF + 1):
        cursor.execute("""UPDATE female SET moyenne_freq_ponderee = ? WHERE id = ?""", (mPondF[i - 1], i))
        cursor.execute("""UPDATE female SET densites = ? WHERE id = ?""", (lo.tabToString(descripteursF[i - 1]), i))

    bdd.commit()
    cursor.close()
    bdd.close()


traitementEnregistrements("bdd_dev.db", 20, 500, 15)
