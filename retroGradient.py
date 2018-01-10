#Package avec toutes les fonctions nécessaires pour effectuer la
#retropropagation du gradient pour UNE SEULE couche cachee


from random import gauss
import data
from math import *
import matplotlib.pyplot as plt
import numpy as np
import listeOperation as lo



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

def g(x,W1,W2):

    #Nombre de variables du modèle
    n = len(x)-1

    #Nombre de neuronnes cachés du modèle (il faut enlever le biais)
    Nc = len(W2)-1

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
    return modele,potentiels





""" Fonction calculant une approximation de la derivee en w du modèle pour l'exemple xk evaluee au point wi
param : x -> exemple prit pour le calcul
        W1 -> poids entre les variables et la couche cachee
        W2 -> poids entre la couche cachee et la sortie 
        h -> pas pour approximer la derivee 
retour : approximation de la derivee du modele au point w donne par W1 et W2 a x fixe"""

def approxg(x,W1,W2,h):
    h1 = [h for i in range(len(W1[0]))]
    h2 = [h for i in range(len(W2))]
    W1plush = []
    W1moinsh = []
    for i in range(len(W1)):
        W1plush.append(lo.add(W1[i],h1))
        W1moinsh.append(lo.sous(W1[i],h1))
    W2plush = lo.add(W2,h2)
    W2moinsh = lo.sous(W2,h2)
    approx = 0.5*(g(x,W1plush,W2plush)[0]-g(x,W1moinsh,W2moinsh)[0])/h
    return approx



"""Fonction calculant l'erreur quadratique moyenne entre deux vecteurs y et g
precondition : y et g sont de même taille"""

def EQM(y,g):
    err = 0
    n = len(g)
    if (n != len(y)):
        print("***EGM***")
        print("len(y) = " + str(len(y)) + " != len(g) = " +str(n))
    for i in range(n):
        err += (y[i] - g[i])**2
    err = 0.5*sqrt(err)
    return err





"""Fonction calculant les "delta" de l'algorithme de retropropagation
param : Nc -> nombre de neurones de la couche cachee
        y -> sortie pour l'exemple considere
        g -> sortie fournie par le reseau pour l'exemple considere
        potentiel -> vecteur des potentiels de chacun des neurones
        W2 -> poids des connexions entre la couche cachee et la sortie
sortie : delta pour les neurones cachés dans les premières cases et celui du neuronne de sortie dans la derniere case
du tableau"""

def retro(Nc,y,g,potentiel,W2):

    #Construction du tableau de derivee : en tout il y a Nc+1 neurones
    delta = [0 for i in range(Nc+1)]

    #Calcul pour le neurone de sortie
    #print("potentiel = " + str(potentiel))
    delta[Nc] = -2*(y-g)*(1/(cosh(potentiel[Nc])**2))

    #Calcul des autres poids
    for i in range(Nc):
        delta[i] = (1/(cosh(potentiel[i])**2))*(delta[Nc]*W2[i+1])

    return delta






""" Fonction effectuant le calcul de la matrice H pour faire la méthode de Levenberg-Marquardt"""

def computeH(invH_prec,zeta):
    transpo = (np.array(zeta)).T
    iH = np.array(invH_prec)
    z = np.array(zeta)
    numerateur = iH*z*transpo*iH
    denominateur = 1 + transpo*iH*z
    invH = np.array(iH) - (numerateur/denominateur)
    return invH.tolist()






""" Fonction calculant le pas de Levenberg-Marquardt pour le pas mu et l'identité comme second terme"""

def pasLM(mu,nbExemples,W1,W2,Z):
    nbPoids = len(W1)*len(W1[0]) + len(W2)
    I = [[0 for i in range(nbPoids)] for j in range(nbPoids)]
    for i in range(nbPoids):
        I[i][i] = (1/mu)
    invH = I
    for k in range(nbExemples):
        zeta = approxg(Z[k],W1,W2,10**(-10))
        invH_prec = invH
        invH = computeH(invH_prec,zeta)
    return invH





""" Fonction modifiant les poids par la méthode de Levenberg (-Marquardt) 
param : W1 -> poids entre les variables et la couche cachee
        W2 -> poids entre la couche cachee et la sortie 
        invH -> matrice inverse intervenant dans le pas de LM
        derW1 -> derivees de la fonction coût par rapport aux param. de la 1ere couche
        derW2 -> derivees de la fonction coût par rapport aux param. de la 2eme couche
"""

def modificationPoids(W1, W2, invH, derW1, derW2):

    #Construction du vecteur des poids
    W = []
    n1 = len(W1)
    n2 = len(W2)

    for i in range(n1):
        for j in range(len(W1[0])):
            W.append([W1[i][j]])
    for i in range(len(W2)):
        W.append([W2[i]])

    #Construction du vecteur gradient de la fonction coût
    derW = []
    for i in range(len(derW1)):
        for j in range(len(derW1[0])):
            derW.append([derW1[i][j]])
    for i in range(len(derW2)):
        derW.append([derW2[i]])

    #Calcul du pas de LM
    pas = np.mat(invH)*(np.mat(derW))

    #Modification des poids
    W = (np.array(W) - pas).tolist()

    Wl = []
    for i in range(len(W)):
        Wl.append(W[i][0])

    #Reconstruction des vecteurs de parametres
    W1_maj = []
    W2_maj = []
    for i in range(n1):
        ligne = []
        for j in range(len(W1[0])):
            ligne.append(Wl[i+j])
        W1_maj.append(ligne)
    for j in range(len(W2)):
        W2_maj.append(Wl[n1+j])

    return W1_maj,W2_maj



"""Fonction effectuant le calcul des parametres du reseau par retropropagation du
gradient pour UNE SEULE couche cachee
param : n -> nombre de variables par exemples (ici nombre d'harmonique par 
             enregistrement
        Nc -> nombre de neurones de LA couche cachée
        seuil -> seuil en deça duquel on s'arrête de minimiser la fonction coût
        l -> pas pour effectuer la modification des poids
        nbIterRechercheMax -> nombre d'itérations maximal pour la méthode de LM
        r -> facteur d'échelle pour mu
retour : tableau des parametres du reseau apres apprentissage"""

def retropropagation(n,Nc,seuil,l,nbIterMax,nbIterRechercheMax,r):

    #Recuperation des spectres et du vecteur de sortie associe
    print("Extraction des données...")
    df = data.DataSet("Data/data",n)
    nbExemples = len(df.index)
    nbHarmoniques = n #df.shape[1]-1;

    Z = []
    y= []

    for i in range(1,nbExemples+1):
        exemple = [1]   #biais de la variable
        ex = df.loc[i]
        for j in range(nbHarmoniques):
            exemple.append(float(ex[j]))
        Z.append(exemple)
        y.append(float(ex[nbHarmoniques]))

    nbExemples = len(Z)

    #Initialisation des parametres
    W1,W2 = initParam(n,Nc)

    #Propagation de tous les exemples avec récuperation des potentiels
    potentielsEx = []
    sortiesEx = []
    for i in range(nbExemples):
        sortie,potentiels = g(Z[i],W1,W2)
        sortiesEx.append(sortie)
        potentielsEx.append(potentiels)

    #Calcul de l'erreur quadratique moyenne associee
    erreur = EQM(y,sortiesEx)

    nbIter = 1

    tableErreur = [erreur]

    #Tant que l'erreur n'est pas suffisamment petite
    while (erreur > seuil) & (nbIter <= nbIterMax):

        #Retropropagation des exemples
        delta = []
        for k in range(nbExemples):
            d = retro(Nc,y[k],sortiesEx[k],potentielsEx[k],W2)
            delta.append(d)
        deriveesW1 = [[0 for j in range(n+1)] for i in range(Nc)]
        deriveesW2 = [0 for i in range(Nc+1)]
        for i in range(Nc):
            for j in range(n+1):
                deriveesW1[i][j] = 0
                for k in range(nbExemples):
                    deriveesW1[i][j] += delta[k][i]*Z[k][j]
        for i in range(Nc+1):
            deriveesW2[i] = 0
            for k in range(nbExemples):
                deriveesW2[i] += delta[k][Nc]*tanh(potentielsEx[k][i])

        ##### MODIFICATION DES POIDS ######
        mu = 0.1
        #Modification de mu (et donc du pas) tant que l'on a pas accepte la modificaiton
        accepte = False
        nbIterRecherche = 1
        while ((not accepte) & (nbIterRecherche <= nbIterRechercheMax)):

            #Calcul de l'inverse du pas du second ordre pour LM (avec l'identité et pas diag(H))
            invH = pasLM(mu,nbExemples,W1,W2,Z)
            W1_prec = W1
            W2_prec = W2
            erreur_prec = erreur

            #Modification des poids avec une constante valant mu
            W1,W2 = modificationPoids(W1,W2,invH,deriveesW1,deriveesW2)

            #Comparaison de l'erreur commise avec ces paramètres et de l'ancienne et decision
            sortiesEx = []
            for i in range(nbExemples):
                sortie,potentiels = g(Z[i],W1,W2)
                sortiesEx.append(sortie)
            erreur = EQM(y,sortiesEx)

            if (erreur < erreur_prec):
                accepte = True
                mu = mu/r
            else :
                mu = mu*r
                W1 = W1_prec
                W2 = W2_prec
                erreur = erreur_prec

            nbIterRecherche += 1

        #Si on est sorti sans trouver de mu convenable, il faut quand même modifier les poids avce la dernière valeur de mu trouvee
        if (nbIterRecherche > nbIterRechercheMax):
            invH = pasLM(mu,nbExemples,W1,W2,Z)
            #Modification des poids avec une constante valant mu
            W1,W2 = modificationPoids(W1,W2,invH,deriveesW1,deriveesW2)

        #Propagation des exemples et recalcule des potentiels
        potentielsEx = []
        sortiesEx = []
        for i in range(nbExemples):
            sortie,potentiels = g(Z[i],W1,W2)
            sortiesEx.append(sortie)
            potentielsEx.append(potentiels)

        #Recalculer l'erreur quadratique moyenne
        erreur = EQM(y,sortiesEx)
        tableErreur.append(erreur)

        nbIter += 1


    #Renvoyer les tableaux des paramètres
    abs = [(i+1) for i in range(len(tableErreur))]
    plt.plot(np.array(abs),np.array(tableErreur))
    plt.show()