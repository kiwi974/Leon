import wave
import os

from matplotlib.pyplot import *
import scipy.io.wavfile as wave
from numpy.fft import fft

# Changement du repertoire de travaille

os.chdir("/home/ray974/Learning/")



""" Fonction ne conservant que les parties entières des éléments d'une liste, et
enlevant ensuite les doublons, tout en maintenant à jour spectre.
param : freq -> liste a traiter 
        spectre -> amplitudes dans le spectre
        freqMin -> plus petite frequence dans le spectre
        freqMax -> plus petite frequence dans le spectre 
        n -> nombre de donnees dans freq
exemple : [2.354,2.648,78.2452,2.47,78.364,78.0] -> [2,78]
"""

def arrDeleteDouble(freq,spectre,n,freqMax):
    for i in range(n):
        freq[i] = freq[i]//1
    deja_vu = [False for i in range(freqMax+2)]
    nFreq = []
    nSpectre = []
    for i in range(n):
        if (not deja_vu[int(freq[i])]):
            nFreq.append(freq[i])
            nSpectre.append(spectre[i])
            deja_vu[int(freq[i])] = True
    return nFreq,nSpectre



def fftFreq(chemin,freqMin,freqMax):

    #Lecture du fichier wav
    rate,data = wave.read(chemin)  #rate = frequence d'echantillonage
    n = data.size
    duree = 1.0*n/rate  #periode du signal

    debut = 0.5

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

    #Traitement des donnees
    n = len(spectre)
    s = []
    for i in range(n):
        s.append(spectre[i][0])

    spectre = s

    freq,spectre = arrDeleteDouble(freq,spectre,n,freqMax)

    return freq,spectre
