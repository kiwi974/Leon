""" Fonctions servant a generer des donnees, pour retrouver ensuite les
distributions qui ont genere ces donnees."""

from random import *

from retroGradient import *


""" Fonction generant un echantillon de donnees gaussiennes autour d'une 
moyenne de variabilite varMoy et d'ecart-type de variabilite varSig"""

def genSampleData(n,moy,varMoy,sigma,varSig):
    sample = []
    somme = 0
    for i in range(n):

        #Ajout d'un bruit sur la moyenne et l'écart-type
        m = moy + random()*varMoy
        s = sigma + gauss(5,3)*varSig

        #Calcul d'une valeur de l'échantillon
        val = gauss(m,s)
        somme += val
        sample.append(val)
    print(somme/n)
    return sample


genSampleData(600000,20,4,7,5)

