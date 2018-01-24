import wave
import os
import csv

from matplotlib.pyplot import *
import scipy.io.wavfile as wave
from numpy.fft import fft

# Changement du repertoire de travaille
os.chdir("/home/ray974/Learning/")





def fftFreq(chemin,freqMin,freqMax):

    #Lecture du fichier wav
    rate,data = wave.read(chemin)  #rate = frequence d'echantillonage
    n = data.size
    duree = 1.0*n/rate  #periode du signal

    debut = 1.5

    #Obtention du spectre
    start = int(debut*rate)
    stop = int((debut+duree)*rate)
    spectre = np.absolute(fft(data[start:stop]))
    n = spectre.size
    freq = np.zeros(n)
    for k in range(n):
        freq[k] = 1.0/n*rate*k

    #Selection des donnees sur le domaine [freqMin,freqMax]
    indFreqMin = 0
    while ((freq[indFreqMin] < freqMin) & (indFreqMin < n)):
        indFreqMin += 1
    indFreqMax = indFreqMin+1
    while ((freq[indFreqMax] < freqMax) & (indFreqMax < n)):
        indFreqMax += 1

    #Extraction des donnees
    freq = freq[indFreqMin:indFreqMax]
    spectre = spectre[indFreqMin:indFreqMax]

    #tracerSpectre(freq,spectre)

    return freq,spectre





def tracerSpectre(freq,spectre):

    spectre = spectre/spectre.max()
    n = len(spectre)

    #Calcul de la moyenne dans ce spectre
    moyf = 0
    moya = 0
    for i in range(n):
        moyf = moyf + freq[i]
        moya = moya + spectre[i]
    moyf = moyf/n
    moya = moya/n

    figure(figsize=(9,4))
    vlines(freq,[0],spectre,'r')
    vlines([moyf],[0],[moya],'b')
    xlabel('f (Hz)')
    ylabel('A')
    #axis([0,0.5*rate,0,1])
    axis([freq[0],freq[n-1],0,1])
    show()





def affichageSpectres():
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
                fftFreq("/media/ray974/common-voice/cv-valid-dev/wav/sample-" + ind + str(nbRow) + ".wav",0,500)
            elif (col[5]=="female"):
                fftFreq("/media/ray974/common-voice/cv-valid-dev/wav/sample-" + ind + str(nbRow) + ".wav",0,500)
            else:
                print('')
            nbRow += 1




#affichageSpectres()