##Package où l'on trouve toutes les fonctions nécessaire pour effectuer la sélection des variables

import math
import time
from random import gauss

import numpy as np

import listeOperation as lo

"""Fonction trouvant une famille libre de dimension N à partir d'un vecteur u
#param : u -> premier vecteur de la famille libre à construire
#        p -> dimension de l'espace dans lequel on recherche une famille libre
#retour : u ainsi que la famille libre de dimension N trouvee
#precondition : N-1 >= 1 i.e. N >= 2"""

def fLibre(u,p):
    famille = [u]
    famille_aux = [u]
    N = len(u)
    b = np.array([0 for i in range(N)])
    libre = 0
    while (libre==0):
        for k in range(p-1):
            v = list(u)
            vectGauss = [gauss(2,5) for i in range(N)]
            v = lo.mult(u,vectGauss)
            famille_aux.append(v)
        #solution = np.linalg.solve(np.transpose(np.array(famille_aux)),b)
        #if (lo.vectNul(abs(solution))):
            #famille = famille_aux[:p]
        libre = 1
        #else:
            #famille_aux = famille
    return famille_aux

#f1 = fLibre([3,2,7,5,2,87],4)
#print(f1)





#Fonction effectuant le procede de GS stable
#param : fLibre -> famille que l'on orthonormalise

def GS(fLibre):
    dim = len(fLibre)
    fOrtho = []

    #Calcul du premier vecteur
    e1 = fLibre[0]
    e1 = lo.div(e1,[math.sqrt(lo.ps(e1,e1)) for i in range(len(e1))])
    fOrtho.append(e1)

    #Calcul des N-1 autres vecteurs
    for k in range(1,dim):
        ek = fLibre[k]
        vk = ek
        for i in range(k):
            ui = fOrtho[i]
            vk = lo.sous(vk,lo.mult([lo.ps(ui,ek) for i in range(len(ui))],ui))
        vk = [i/(math.sqrt(lo.ps(vk,vk))) for i in vk]
        fOrtho.append(vk)
    return fOrtho

#gs1 = GS(f1)
#print(gs1)
#print(lo.sysLin(gs1,[0,0,0]))
#print(lo.orthonormee(gs1))






#Fonction qui calcul la projection du vecteur u sur la famille de vecteurs b
#param : u -> vecteur dont on cherche la projection
#        b -> famille sur laquelle on projete

def projection(u,b):
    dim = len(b)
    projete = [0 for i in range(dim)]
    for i in range(dim):
        projete[i] = lo.ps(u,b[i])
    return projete

#proj = projection([1,2,3],gs1)
#print(proj)






#Fonction qui projete les vecteurs de la famille f sur ceux de la famille b
#param : f -> famille de vecteurs dont on cherche la projection sur b
#        b -> famille de vecteurs sur laquelle on projete

def projFamille(f,b):
    dim = len(f)
    nEspace = []
    for i in range(dim):
        v = projection(f[i],b)
        nEspace.append(v)
    return nEspace

#projF = projFamille([[1,2,3],[4,5,6],[7,8,9]],gs1[1:])
#print(projF)


#Fonction qui, étant donné un vecteur u, et une liste de vecteur f, projète
#tous les vecteurs de f sur le sous-espace orthogonal a u
#param : u -> vecteur tel que l'on projete sur son orthogonal Ort
#        f -> famille dont on cherche les composantes dans Ort pour chaque vecteur
#        y -> autre vecteur que l'on souhaite projeter sur cette famille
#        p -> dimension de l'espace des observations courant

def projSEOrtho(u,f,p):
    fLibreU = fLibre(u,p)
    fOrtho = GS(fLibreU)
    #Il faut retirer le premier vecteur de cette liste, car il s'agit de la composante relative a u
    #fOrtho = fOrtho[1:]
    fOrtho[0] = [0 for i in range(len(fOrtho[0]))]
    nFamille = projFamille(f,fOrtho)
    return nFamille

#projSEO = projSEOrtho([1,2,3],[[1,2,3],[4,5,6],[7,8,9]],[7,5,3],3)
#print(projSEO)


#Fonction calculant le coefficient de correlation entre deux vecteurs u et v centrés
#precondition : u et v sont de meme dimension

def corr(u,v):
    ps1 = lo.ps(u,v)
    ps2 = lo.ps(u,u)
    ps3 = lo.ps(v,v)
    if (ps3==0):
        return 0
    else :
        corr = ps1**2/(ps2*ps3)
        return corr





#Fonction engendrant une variable sonde de dimension N à partir d'une distribution de
#moyenne nulle et de variance 1

def genSonde(N):
    sonde = []
    for i in range(N):
        sonde.append(gauss(0,1))
    return sonde

#sonde1 = genSonde(5)
#print(sonde1)





#Fonction qui effectue le classement de variables candidates
#param : y -> vecteur des observations
#        variables -> vecteur contenant toutes les variables candidates

def classer(y,variables,sonde):
    N = len(variables[0])
    dimEspVar = len(variables[0])
    pvar = list(variables)
    pvar.append(sonde)
    varClassement = []
    #py = list(y)
    p = len(pvar)
    ordreVar = [(i+1) for i in range(p)]

    #Boolen valant true ssi la variable classee est la variable sonde
    rencontree = 0

    #Liste ou l'on stocke les variables rencontrees
    ordre = []

    #Iteration du procede
    while (N > 1) & (p > 1) & (rencontree == 0):
        #Calcul des coefficients de correlation
        coeffCorr = []
        for i in range(p):
            c = corr(y,pvar[i])
            coeffCorr.append(c)

        #Obtention du vecteur le plus corrélé à y
        indMax = coeffCorr.index(max(coeffCorr))
        u = pvar[indMax]
        ordre.append(u)
        varClassement.append(ordreVar[indMax])

        #Si c'est la variable sonde, on arrête le processus
        if (indMax == (p-1)):
            #print("On est tombé sur la variable sonde!")
            rencontree+=1
            break

        #Suppression de cette variable dans la liste des variables
        del pvar[indMax]
        del ordreVar[indMax]
        p-=1

        #Projection de y et toutes les variables non sélectionnées sur le sous-espace
        #orthogonal à u
        pvar = projSEOrtho(u,pvar,dimEspVar)

        N-=1

    # N = 1 : il reste une variable à classer
    print(varClassement)
    return ordre

#y1 = [2,3,5]
#variables1 = [[5,4,2],[4,8,7],[3,1,2],[7,2,3],[4,6,2]]
#print(classer(y1,variables1,genSonde(3)))






#Fonction trouvant la distribution de probabilité empirique des variables non pertinentes
#en réalisant le classement avec nbVarSonde réalisation de la variable sonde (gaussienne)
def distriNonPertinentes(y,variables,nbVarSonde):

    debut = time.time()

    N = len(variables[0])
    p = len(variables)

    distri = [0 for i in range(p+1)]

    #Generation de nbVarSonde variables sondes
    sondes = [genSonde(N) for i in range(nbVarSonde)]

    #Pour ces sondes, on effectue le classement et on récupère la place de la sonde
    for i in range(nbVarSonde):
        ordre = classer(y,variables,sondes[i])
        place = len(ordre)
        print("La sonde n° " + str(i) + " a été classé au rang " + str(len(ordre)) + ".")
        distri[place]+=1

    fin = time.time()
    print("Le classement des variables sonde a prit " + str(fin-debut) + " secondes.")

    return distri

#variables2 = [[5,4,2],[4,8,7],[3,1,2],[7,2,3],[4,6,2],[8,4,2],[7,5,2],[6,3,1],[3,3,3],
#              [8,8,7],[7,4,4],[2,3,6],[9,5,4],[11,1,1],[54,2,1],[7,1,56],[6,6,9],[11,11,11]]
#d = distriNonPertinentes(y1,variables2)
#print(d)