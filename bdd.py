import sqlite3
import os
import csv
import FFT
import listeOperation as lo


#Changement du repertoire de travaille
os.chdir("/home/ray974/Learning/Data")



def initaliazeBDD(name):

    bdd = sqlite3.connect(name)

    cursor = bdd.cursor()

    #Creation d'un table homme
    cursor.execute("""
          CREATE TABLE IF NOT EXISTS male(
          id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
          nom_extrait TEXT,
          frequences TEXT, 
          amplitudes TEXT
          )
    """)

    #Creation d'un table femme
    cursor.execute("""
          CREATE TABLE IF NOT EXISTS female(
          id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
          nom_extrait TEXT,
          frequences TEXT, 
          amplitudes TEXT
          )
    """)
    bdd.commit()

    bdd.close()


initaliazeBDD("bdd_dev.db")


def traitementEnregistrements(name,freqMin,freqMax):

    bdd = sqlite3.connect(name)
    cursor = bdd.cursor()

    with open("/media/ray974/common-voice/cv-valid-dev.csv", 'rt') as f:
        reader = csv.reader(f)
        nbRow = 0
        for col in reader:
            if (nbRow >= 50):
                break
            ind = ""
            for i in range(6-len(str(nbRow))):
                if (ind == ""):
                    ind = "0"
                else:
                    ind = ind + "0"
            if (col[5]=="male"):
                freq,spectre = FFT.fftFreq("/media/ray974/common-voice/cv-valid-dev/wav/sample-" + ind + str(nbRow) + ".wav", freqMin, freqMax)
                freq = lo.tabToString(freq)
                spectre = lo.tabToString(spectre)
                chaine = "sample-" + ind + str(nbRow)
                cursor.execute("INSERT INTO male(nom_extrait, frequences, amplitudes) VALUES(?,?,?)", (chaine, freq, spectre))
            elif (col[5]=="female"):
                freq, spectre = FFT.fftFreq("/media/ray974/common-voice/cv-valid-dev/wav/sample-" + ind + str(nbRow) + ".wav",freqMin,freqMax)
                freq = lo.tabToString(freq)
                spectre = lo.tabToString(spectre)
                chaine = "sample-" + ind + str(nbRow)
                cursor.execute("INSERT INTO female(nom_extrait, frequences, amplitudes) VALUES(?,?,?)", (chaine, freq, spectre))
            else:
                useless = 1
            nbRow += 1

    bdd.commit()
    cursor.close()
    bdd.close()


traitementEnregistrements("bdd_dev.db")