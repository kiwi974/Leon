import numpy as np

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






