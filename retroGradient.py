#Package avec toutes les fonctions nécessaires pour effectuer la
#retropropagation du gradient pour UNE SEULE couche cachee


from random import gauss
import data
from math import *


"""Fonction initialisant la matrice des parametres (poids synaptiques)
param : n -> nombre de variables par exemples (ici nombre d'harmonique par 
             enregistrement
        Nc -> nombre de neurones de LA couche cachée
retour : W1 -> matrice des parametres entre l'entree et la couche cachee
         W2 -> vecteur des parametres entre la couche cachee et la sortie"""

def initParam(n,Nc):

    #Initialisation de W1 et W2, avec des distri centrees de variance n-1 pour
    #ne pas saturer les sigmoïdes (les potentiels des neurones cachés ont une
    #variance de 1
    W1 = [[gauss(0,1/(n-1)) for k in range(n+1)] for i in range(Nc)]
    W2 = [gauss(0,1/(n-1)) for i in range(Nc+1)]

    #PLes paramètres relatifs au biais doivent être nuls
    for i in range(Nc):
        W1[i][0] = 0
    W2[0] = 0

    return W1,W2





"""Fonction calculant la sortie d'un reseau de n variables (+un biais), UNE SEULE couche
cachee et UNE SEULE sortie
param : x -> exemple prit pour le calcul
        W1 -> poids entre les variables et la couche cachee
        W2 -> poids entre la couche cachee et la sortie 
sortie : prediction du modele pour l'exemple x et tableau des potentiels des 
        differents neurones du reseau"""

def g(x,W1,W2,n,Nc):
    modele = W2[0]
    #Le tableau des potentiels est initialisé avec le biais de la couche cachee
    potentiels = [1]
    for i in range(0,Nc):
        #Calcul du potentiel du neurone i
        p = W1[i][0]
        for j in range(1,n+1):
            p += W1[i][j]*x[j-1]
        potentiels.append(p)
        modele += W2[i]*tanh(p)
    potentiels.append(modele)
    return modele,potentiels





"""Fonction calculant l'erreur quadratique moyenne entre deux vecteurs y et g
precondition : y et g sont de même taille"""

def EQM(y,g):
    err = 0
    n = len(g)
    for i in range(n):
        err += (y[i] - g[i])**2
    err = 0.5*math.sqrt(err)
    return err





"""Fonction calculant les "delta" de l'algorithme de retropropagation
param : Nc -> nombre de neurones de la couche cachee
        y -> vecteur des sorties attendues pour les differents exemples
        g -> vecteur des sorties du modele pour les differents exemples
        potentiel -> vecteur des potentiels que chacun des neurones
        W2 -> poids des connexions entre la couche cachee et la sortie"""

def retro(Nc,y,g,potentiel,W2):

    #Construction du tableau de derivee : en tout il y a Nc+1 neurones
    delta = [0 for i in range(Nc+2)]

    #Calcul pour le neurone de sortie
    delta[Nc+1] = -2*(y-g)*(1/(cosh(potentiel[Nc+1])**2))

    #Calcul des autres poids
    for i in range(Nc+1):
        delta[i] = (1/(cosh(potentiel[i])**2))*(delta[Nc]*W2[i+1])

    return delta





"""Fonction effectuant le calcul des parametres du reseau par retropropagation du 
gradient pour UNE SEULE couche cachee
param : n -> nombre de variables par exemples (ici nombre d'harmonique par 
             enregistrement
        Nc -> nombre de neurones de LA couche cachée
        seuil -> seuil en deça duquel on s'arrête de minimiser la fonction coût
        l -> pas pour effectuer la modification des poids
retour : tableau des parametres du reseau apres apprentissage"""

def retropropagation(n,Nc,seuil,l,nbIterMax):

    #Initialisatiuon des exemples et du vecteurs de sortie avec 10 harmonique
    y,Z = data.dataSet(10)
    nbExemples = len(Z)

    #Initialisation des parametres
    W1,W2 = initParam(n,Nc)

    #Propagation de tous les exemples avec récuperation des potentiels
    potentielsEx = []
    sortiesEx = []
    for i in range(nbExemples):
        sortie,potentiels = g(Z[i],W1,W2,n,Nc)
        sortiesEx.append(sortie)
        potentielsEx.append(potentiels)

    #Calcul de l'erreur quadratique moyenne associee
    erreur = EQM(y,sortiesEx)

    numEx = 0
    nbIter = 1

    #Tant que l'erreur n'est pas suffisamment petite
    while (erreur > seuil) & (nbIter <= nbIterMax):

        #Retropropagation de l'exemple numero numEx
        delta = retro(Nc,y,sortiesEx,potentielsEx[numEx],W2)
        deriveesW1 = [[0 for j in range(n+1)] for i in range(Nc)]
        deriveesW2 = [0 for i in range(Nc+1)]
        for i in range(Nc):
            for j in range(n+1):
                deriveesW1[i][j] = delta[i]*Z[numEx][j]
        for i in range(Nc+1):
            deriveesW2[i] = delta[Nc+1]*tanh(potentielsEx[numEx][i])

        #Modification des poids
        for i in range(Nc):
            for j in range(n+1):
                W1[i][j] = W1[i][j] - (l/nbIter)*deriveesW1[i][j]
        for i in range(Nc+1):
            W2[i] = W2[i] - (l/nbIter)*deriveesW2[i]

        #Propagation des exemples pour recalculer les potentiels
        potentielsEx = []
        sortiesEx = []
        for i in range(nbExemples):
            sortie,potentiels = g(Z[i],W1,W2,n,Nc)
            sortiesEx.append(sortie)
            potentielsEx.append(potentiels)

        #Recalculer l'erreur quadratique moyenne
        erreur = EQM(y,sortiesEx)

        #Incrementer numEx
        numEx = (numEx+1)//nbExemples

    #Renvoyer les tableaux des paramètres