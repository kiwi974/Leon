#Package dont les fonctions serviront au traitement des data qui seront utilisee
#pour l'apprentissage


import FFT
import math
import listeOperation as lo


fichierH = ["test_Leon","test_Loic","Raymond","test_Walh","test_Philippe","test_Romain","test_Mathieu","test_JP"]
fichierF = ["test_maman","test_Alex","test_Joanne","test_Victime","test_VictimeSoeur","test_Cathy","test_Justine","test_Delphine"]





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


"""Fonction qui contruit les exemples d'apprentissage ainsi que le vecteur de sortie
y (similaire à constructVA mais construit les exemples, non les variables"""

def dataSet(nbHarmoniques):

    #Vecteur des differentes realisations des variables
    Z = []

    #Vecteur des mesures : 1 pour les hommes et -1 pour les femmes
    y = []

    #Acquisition des donnees concernant les hommes
    print("Hommes")
    for k in range(len(fichierH)):
        spectre = FFT.fftFreq("/home/ray974/Learning/VoiceRecord/homme/"+fichierH[k]+".wav",nbHarmoniques)
        #print(spectre)
        Z.append(spectre)
        y.append(1)

    print("Femmes")
    #Acquisition des donnees concernant les femmes
    for k in range(len(fichierF)):
        spectre = FFT.fftFreq("/home/ray974/Learning/VoiceRecord/femme/"+fichierF[k]+".wav",nbHarmoniques)
        #print(spectre)
        Z.append(spectre)
        y.append(-1)

    #Pretraitement des donnees
    print("Pretraitement")
    for k in range(len(Z)):
        Z[k] = pretraitement(Z[k])

    return y,Z