#Package avec toutes les fonctions nécessaires pour effectuer la
#retropropagation du gradient pour UNE SEULE couche cachee


from random import gauss
from math import *
import numpy as np
import listeOperation as lo
import exploitation as explo

""" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!  On prendra n et Nc en parametres a chaque fois que necessaire !!!!!
    !!!!!!  car ils seront connus dans l'algo de retropropagation       !!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""




"""Fonction initialisant la matrice des parametres (poids synaptiques)
param : n -> nombre de variables 
        Nc -> nombre de neurones de LA couche cachée
retour : W -> vecteur des paramètres du réseau. """

def initParam(n,Nc):
    W = []
    #Connexions entre les variables et la premiere couche
    for i in range(Nc):
        #Initialement la composante du biais est nulle
        W.append(0)
        for j in range(1,n+1):
            W.append(gauss(0,1/(n-1)))
    #Connexions entre la dernière couche et la sortie
    W.append(0)  #Composante nulle selon le biais en sortie
    for j in range(1,Nc+1):
        W.append(gauss(0,1/(n-1)))
    return W

#print(initParam(2,3)[9])





""" Fonction calculant la sortie d'un reseau de n variables (+un biais), UNE SEULE couche
cachee et UNE SEULE sortie
param : x -> exemple prit pour le calcul
        n -> nombre de variables par exemples (ici nombre d'harmonique par 
             enregistrement
        Nc -> nombre de neurones de LA couche cachée
        W1 -> poids entre les variables et la couche cachee
        W2 -> poids entre la couche cachee et la sortie 
sortie : prediction du modele pour l'exemple x et tableau des potentiels des 
        differents neurones du reseau """

def g(x,n,Nc,W):
    modele = W[(n+1)*Nc]
    #Le tableau des potentiels est initialisé avec le biais de la couche cachee
    potentiels = [1]
    for i in range(Nc):
        #Calcul du potentiel du neurone i
        p = 0
        for j in range(n+1):
            p += W[i*(n+1)+j]*x[j]
        potentiels.append(p)
        modele += W[Nc*(n+1)+i]*tanh(p)
    return modele,potentiels





""" Fonction calculant l'erreur quadratique moyenne entre deux vecteurs y et g
precondition : y et g sont de même taille. """

def EQM(y,g):
    err = 0
    errk = []
    n = len(g)
    if (n != len(y)):
        print("***EQM***")
        print("len(y) = " + str(len(y)) + " != len(g) = " +str(n))
    for i in range(n):
        r = y[i] - g[i]
        errk.append(r)
        err += r**2
    err = 0.5*sqrt(err)
    return err, errk





""" Fonction calculant les "delta" de l'algorithme de retropropagation.
param : Nc -> nombre de neurones de la couche cachee
        y -> sortie pour l'exemple considere
        g -> sortie fournie par le reseau pour l'exemple considere
        potentiel -> vecteur des potentiels de chacun des neurones
        W2 -> poids des connexions entre la couche cachee et la sortie
sortie : delta pour les neurones cachés dans les premières cases et celui du neuronne de sortie dans la derniere case
du tableau """

def retro(n,Nc,y,g,potentiel,W):

    #Construction du tableau de derivee : en tout il y a Nc+1 neurones
    delta = []
    #Calcul pour le neurone de sortie
    #print("potentiel = " + str(potentiel))
    deltaNc = -2*(y-g)*(1/(cosh(potentiel[Nc])**2))

    #Calcul des autres poids
    for i in range(Nc+1):
        der = (1/(cosh(potentiel[i])**2))*(deltaNc*W[(n+1)*Nc+i])
        delta.append(der)

    #Ajout de la derivee de la sortie
    delta.append(deltaNc)

    return delta





""" Fonction calculant la derivee en w du modèle pour l'exemple xk evaluee au point W
param : err -> erreur associee à l'exemple k
        derivPi -> derivees des fonctions coût par rapport à chaque variable
retour : approximation de la derivee du modele au point w donne par W1 et W2 a x fixe"""

def derg(err,derivPi,nbPoids):
    derivg = []
    for i in range(nbPoids):
        derivg.append(derivPi/(-2*err))
    return derivg




""" Fonction effectuant le calcul de la matrice H pour faire la méthode de Levenberg-Marquardt"""

def computeH(invH_prec,zeta):
    transpo = (np.array(zeta)).T
    iH = np.array(invH_prec)
    z = np.array(zeta)
    numerateur = iH*z*transpo*iH
    denominateur = 1 + transpo*iH*z
    invH = np.array(iH) - (numerateur/denominateur)
    return invH.tolist()






""" Fonction calculant le pas de Levenberg-Marquardt pour le pas mu et l'identité comme second 
terme. """

def pasLM(nbPoids,mu,nbExemples,deriveesg):
    I = [[0 for i in range(nbPoids)] for j in range(nbPoids)]
    for i in range(nbPoids):
        I[i][i] = (1/mu)
    invH = I
    for k in range(nbExemples):
        zeta = deriveesg[k]
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

def modificationPoids(W, invH, derW):

    nW = []

    for i in range(len(invH)):
        nW.append(lo.ps(invH[i],derW))
    W = list(nW)
    return W





""" Fonction calculant le gradient au point représenté par W. """

def gradient(nbExemples,n,Nc,y,sortiesEx,potentielsEx,Z,W):
    delta = []
    derg = [[] for i in range(nbExemples)]
    for k in range(nbExemples):
        d = retro(n,Nc,y[k],sortiesEx[k],potentielsEx[k],W)
        delta.append(d)
    deriveesW = []
    for i in range(1,Nc+1):
        for j in range(n+1):
            der = 0
            for k in range(nbExemples):
                d = delta[k][i]*Z[k][j]
                if (sortiesEx[k] == 0.0):
                    derg[k].append(0.0)
                else :
                    derg[k].append(d/(-2*sortiesEx[k]))
                der += d
            deriveesW.append(der)
    for i in range(Nc+1):
        der = 0
        for k in range(nbExemples):
            d = delta[k][Nc+1]*tanh(potentielsEx[k][i])
            if (sortiesEx[k] == 0.0):
                derg[k].append(0.0)
            else :
                derg[k].append(d/(-2*sortiesEx[k]))
            der += d
        deriveesW.append(der)

    return deriveesW, derg





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

    #Liste des variables créées
    var = []

    #Tableau flag pour la recherche du pas dans Levengerb
    flag = [0,0]

    #Recuperation des spectres et du vecteur de sortie associe
    if (y == []): #on suppose implicitement qu'alors Z est également vide (on ne précise pas l'un sans preciser l'autre)
        y,Z,nbDesc = explo.getDataEx(chemin)
        var.append(y)
        var.append(Z)

    #Initialisation des parametres
    W = initParam(n,Nc)
    nbPoids = (n+1)*Nc + Nc+1 #=len(W)
    var.append(W)

    nbExemples = len(Z)
    var.append(nbExemples)

    #Propagation de tous les exemples avec récuperation des potentiels
    potentielsEx = []
    sortiesEx = []
    var.append(potentielsEx)
    var.append(sortiesEx)
    for i in range(nbExemples):
        sortie,potentiels = g(Z[i],n,Nc,W)
        sortiesEx.append(sortie)
        potentielsEx.append(potentiels)

    #Calcul de l'erreur quadratique moyenne associee
    erreur,errk = EQM(y,sortiesEx)
    var.append(erreur)
    var.append(errk)

    #Norme du gradient au point de depart
    deriveesW, deriveesg = gradient(nbExemples,n,Nc,y,sortiesEx,potentielsEx,Z,W)
    var.append(deriveesW)
    normG0 = lo.norm2(deriveesW)
    #print("nomrG0 = " + str(normG0))
    normG = 10**(10)

    nbIter = 1
    tableErreur = [erreur]

    #Tant que l'erreur n'est pas suffisamment petite
    while (normG > c1*normG0) & (nbIter <= nbIterMax) & (erreur > seuil):

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
            invH = pasLM(nbPoids,mu,nbExemples,deriveesg)
            if (nbIter ==1) & (nbIterRecherche==1):
                var.append(invH)
            W_prec = W
            erreur_prec = erreur

            #Modification des poids avec une constante valant mu
            W = modificationPoids(W,invH,deriveesW)

            #Comparaison de l'erreur commise avec ces paramètres et de l'ancienne et decision
            sortiesEx = []
            for i in range(nbExemples):
                sortie,potentiels = g(Z[i],n,Nc,W)
                sortiesEx.append(sortie)
            erreur,errk = EQM(y,sortiesEx)

            if (erreur < erreur_prec):
                accepte = True
                mu = mu/r
                flag[0] += 1
            else :
                mu_calc.append(mu)
                erreur_calc.append(erreur)
                mu = mu*r
                W = W_prec
                erreur = erreur_prec

            nbIterRecherche += 1

        #Si on est sorti sans trouver de mu convenable, il faut quand même modifier les poids avce la dernière valeur de mu trouvee
        if (nbIterRecherche > nbIterRechercheMax) & (not accepte):
            #On regarde quelle erreur était la plus petite parmis celles recontrees, et on recalcule les poids en conséquence
            indErrMin = lo.indMin(erreur_calc)
            mu = mu_calc[indErrMin]
            invH = pasLM(nbPoids,mu,nbExemples,deriveesg)
            #Modification des poids avec une constante valant mu
            W = modificationPoids(W,invH,deriveesW)
            flag[1] += 1

        #Propagation des exemples et recalcule des potentiels
        potentielsEx = []
        sortiesEx = []
        for i in range(nbExemples):
            sortie,potentiels = g(Z[i],n,Nc,W)
            sortiesEx.append(sortie)
            potentielsEx.append(potentiels)

        #Recalculer l'erreur quadratique moyenne
        erreur,errk = EQM(y,sortiesEx)
        tableErreur.append(erreur)

        #Retropropagation des exemples
        deriveesW, deriveesg = gradient(nbExemples,n,Nc,y,sortiesEx,potentielsEx,Z,W)
        normG = lo.norm2(W)

        nbIter += 1

    print("              " +str(flag))

    #Nettoyage memoire
    explo.clear(var)

    #Trace
    return tableErreur

