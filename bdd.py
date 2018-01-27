import sqlite3
import os
import csv
import FFT
import listeOperation as lo
import exploitation as explo


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
          amplitudes TEXT,
          moyenne_freq_ponderee REAL,
          densites TEXT
          )
    """)

    #Creation d'un table femme
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









def traitementEnregistrements(name,freqMin,freqMax):

    bdd = sqlite3.connect(name)
    cursor = bdd.cursor()

    nbFiles = len(os.listdir("/media/ray974/common-voice/cv-valid-dev/wav"))

    descripteursH = []
    descripteursF = []

    nbH = 0
    nbF = 0


    with open("/media/ray974/common-voice/cv-valid-dev.csv", 'rt') as f:
        reader = csv.reader(f)
        nbRow = 0
        for col in reader:
            if (nbRow > nbFiles-1):
                break
            ind = ""
            for i in range(6-len(str(nbRow))):
                if (ind == ""):
                    ind = "0"
                else:
                    ind = ind + "0"
            if (col[5]=="male"):
                freq,spectre = FFT.fftFreq("/media/ray974/common-voice/cv-valid-dev/wav/sample-" + ind + str(nbRow) + ".wav", freqMin, freqMax)
                n = len(freq)
                mPond = explo.pondMoy(freq,spectre,n)
                l = (explo.densite(spectre,n)).append(mPond)
                descripteursH.append(l)
                freq = lo.tabToString(freq)
                spectre = lo.tabToString(spectre)
                chaine = "sample-" + ind + str(nbRow)
                cursor.execute("INSERT INTO male(nom_extrait, frequences, amplitudes) VALUES(?,?,?)", (chaine, freq, spectre))
                nbH += 1
            elif (col[5]=="female"):
                freq, spectre = FFT.fftFreq("/media/ray974/common-voice/cv-valid-dev/wav/sample-" + ind + str(nbRow) + ".wav",freqMin,freqMax)
                n = len(freq)
                mPond = explo.pondMoy(freq,spectre,n)
                l = (explo.densite(spectre,n)).append(mPond)
                descripteursF.append(l)
                freq = lo.tabToString(freq)
                spectre = lo.tabToString(spectre)
                chaine = "sample-" + ind + str(nbRow)
                cursor.execute("INSERT INTO female(nom_extrait, frequences, amplitudes) VALUES(?,?,?)", (chaine, freq, spectre))
                nbF += 1
            else:
                useless = 1
            nbRow += 1
    print(nbH)
    print(nbF)

    bdd.commit()
    cursor.close()
    bdd.close()


traitementEnregistrements("bdd_dev.db",20,500)