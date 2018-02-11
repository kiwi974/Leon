import numpy as np
from math import *

#MAddition entre deux éléments d'une liste deux à deux
#precondition : u et v sont de même taille

def add(u,v):
    res = list(u)
    for i in range(len(u)):
        res[i] = u[i]+v[i]
    return res





#Soustraction entre deux éléments d'une liste deux à deux
#precondition : u et v sont de même taille

def sous(u,v):
    res = list(u)
    for i in range(len(u)):
        res[i] = u[i] - v[i]
    return res





#Multiplication entre deux éléments d'une liste deux à deux
#precondition : u et v sont de même taille

def mult(u,v):
    res = list(u)
    for i in range(len(u)):
        res[i] = u[i]*v[i]
    return res





#Division entre deux éléments d'une liste deux à deux
#precondition : u et v sont de même taille
#retour w tq w[i] = u[i]/v[i]

def div(u,v):
    res = list(u)
    for i in range(len(u)):
        res[i] = u[i]/v[i]
    return res





#Fonction resolvant un systeme lineaire à partir de listes
#param : a -> matrice rangee ligne par ligne dans des listes
#        b -> partie droite de l'égalite rangee dans un vecteur ligne (liste)
def sysLin(a,b):
    x = np.linalg.solve(np.array(a), np.array(b))
    return x





#Fonction qui calcul le produit scalaire entre deux vecteurs u et v

def ps(u,v):
    res = 0
    dim = len(u)
    for i in range(dim):
        res  = res + u[i]*v[i]
    return res





#Fonction renvoyant true ssi les vecteurs de b forment une famille orthonormee
def orthonormee(b):
    ortho = True
    normee = True
    n = len(b)
    for i in range(n):
        for j in range(i,n):
            if (i==j):
                normee = normee*((ps(b[i],b[j])-1) < 10**(-10))
            else:
                ortho = ortho*(ps(b[i],b[j]) < 10**(-10))
    return ortho*normee





#Fonction testant si un vecteur est numériquement nul
def vectNul(v):
    nul = True
    i = 0
    while (nul) & (i<len(v)):
        nul = nul*(v[i]<10**(-15))
        i+=1
    return nul





""" Fonction renvoyant un représentation d'une liste sous forme d'un string 
param : l -> la liste a traiter
exemple : [2,5,8] --> 2<->5<->8 """

def tabToString(l):
    n = len(l)
    chaine = str(l[0])
    for i in range(1,n):
        chaine = chaine + "<->" + str(l[i])
    return chaine





""" Fonction trouvant l'indice du minimum d'une liste. """

def indMin(l):
    ind = 0
    min = l[0]
    for i in range(1,len(l)):
        if (l[i] < min):
            min = l[i]
            ind = i
    return ind

#print(str(indMin([2,4,8])) + str(indMin([48,42,8]))  + str(indMin([2,-3,8])))





""" Fonction calculant la norme euclidienne d'un vecteur. """

def norm2(v):
    norm = 0
    for i in range(len(v)):
        norm += v[i]**2
    norm = sqrt(norm)
    return norm





""" Fonction qui transpose un vecteir colonne. """
def transpose(l):
    t = []
    for u in l:
        t.append([u])
    return t





""" Fonction calculant la répartition des élements d'une liste selon leur distance à 1 ou -1. """

def distriDistUn(y,l):
    distri = [0 for i in range(11)]
    procheZero = 0
    for i in range(len(l)):
        d = 1
        trouve = False
        while (d <= 10) & (not trouve):
            if (abs(y[i]-l[i]) <= d/10):
                distri[d-1] += 1
                trouve = True
            d += 1
        if (abs(y[i]-l[i]) > 1):
            distri[10] += 1
        if (abs(l[i]) <= 10**(-6)):
            procheZero += 1
    distri.append(str(procheZero))
    return distri

""" Somme des éléments d'un tableau. """

def sum(l):
    s = 0
    for i in range(len(l)):
        s += l[i]
    return s