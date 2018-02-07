#Package avec toutes les fonctions nécessaires pour effectuer la
#retropropagation du gradient pour UNE SEULE couche cachee


from random import gauss
import data
from math import *
import matplotlib.pyplot as plt
import numpy as np
import listeOperation as lo
import exploitation as explo



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
        print("***EQM***")
        print("len(y) = " + str(len(y)) + " != len(g) = " +str(n))
    for i in range(n):
        r = y[i] - g[i]
        err += r**2
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





""" Fonction calculant la norme de W tel que représenté par deux matrices W1 et W2. """

def normW(W1,W2):
    #Construction du vecteur des poids
    W = []
    n1 = len(W1)
    n2 = len(W2)

    for i in range(n1):
        for j in range(len(W1[0])):
            W.append(W1[i][j])
    for i in range(n2):
        W.append(W2[i])

    return lo.norm2(W)





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
    for i in range(n2):
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





""" Fonction calculant le gradient au point représenté par W1 et W2. """

def gradient(nbExemples,n,Nc,y,sortiesEx,potentielsEx,Z,W2):
    delta = []
    for k in range(nbExemples):
        d = retro(Nc,y[k],sortiesEx[k],potentielsEx[k],W2)
        delta.append(d)
    deriveesW1 = [[0 for j in range(n+1)] for i in range(Nc)]
    deriveesW2 = [0 for i in range(Nc+1)]
    for i in range(Nc):
        for j in range(n):
            deriveesW1[i][j] = 0
            for k in range(nbExemples):
                deriveesW1[i][j] += delta[k][i]*Z[k][j]
    for i in range(Nc+1):
        deriveesW2[i] = 0
        for k in range(nbExemples):
            deriveesW2[i] += delta[k][Nc]*tanh(potentielsEx[k][i])
    return deriveesW1,deriveesW2





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

def retropropagation(chemin,n,Nc,seuil,c1,mu_0,nbIterMax,nbIterRechercheMax,r,y=[],Z=[]):

    #Tableau flag pour la recherche du pas dans Levengerb
    flag = [0,0]

    #Recuperation des spectres et du vecteur de sortie associe
    if (y == []): #on suppose implicitement qu'alors Z est également vide (on ne précise pas l'un sans preciser l'autre)
        y,Z,nbDesc = explo.getDataEx(chemin)

    #Initialisation des parametres
    W1,W2 = initParam(n,Nc)

    nbExemples = len(Z)

    #Propagation de tous les exemples avec récuperation des potentiels
    potentielsEx = []
    sortiesEx = []
    for i in range(nbExemples):
        sortie,potentiels = g(Z[i],W1,W2)
        sortiesEx.append(sortie)
        potentielsEx.append(potentiels)

    #Calcul de l'erreur quadratique moyenne associee
    erreur = EQM(y,sortiesEx)

    #Norme du gradient au point de depart
    deriveesW1, deriveesW2 = gradient(nbExemples,n,Nc,y,sortiesEx,potentielsEx,Z,W2)
    normG0 = normW(deriveesW1,deriveesW2)
    print("nomrG0 = " + str(normG0))
    normG = 10**(10)

    nbIter = 1
    tableErreur = [erreur]

    #Tant que l'erreur n'est pas suffisamment petite
    while (normG > c1*normG0) & (nbIter <= nbIterMax) & (erreur > seuil):

        print(nbIter)

        ##### MODIFICATION DES POIDS ######
        mu = mu_0
        #Modification de mu (et donc du pas) tant que l'on a pas accepte la modification
        accepte = False
        nbIterRecherche = 1
        #Tableaux pour retenir les mu et les erreurs associées
        mu_calc = []
        erreur_calc = []
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
                flag[0] += 1
            else :
                mu_calc.append(mu)
                erreur_calc.append(erreur)
                mu = mu*r
                W1 = W1_prec
                W2 = W2_prec
                erreur = erreur_prec

            nbIterRecherche += 1

        #Si on est sorti sans trouver de mu convenable, il faut quand même modifier les poids avce la dernière valeur de mu trouvee
        if (nbIterRecherche > nbIterRechercheMax) & (not accepte):
            #On regarde quelle erreur était la plus petite parmis celles recontrees, et on recalcule les poids en conséquence
            indErrMin = lo.indMin(erreur_calc)
            mu = mu_calc[indErrMin]
            invH = pasLM(mu,nbExemples,W1,W2,Z)
            #Modification des poids avec une constante valant mu
            W1,W2 = modificationPoids(W1,W2,invH,deriveesW1,deriveesW2)
            flag[1] += 1

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

        #Retropropagation des exemples
        deriveesW1, deriveesW2 = gradient(nbExemples,n,Nc,y,sortiesEx,potentielsEx,Z,W2)
        normG = normW(deriveesW1,deriveesW2)

        nbIter += 1

    print(flag)

    #Renvoyer les tableaux des paramètres
    abs = [(i+1) for i in range(len(tableErreur))]
    plt.plot(np.array(abs),np.array(tableErreur))
    plt.show()


#retropropagation("/home/ray974/Learning/Data/bdd_dev.db",15,3,1.5,2,20,100,10)
