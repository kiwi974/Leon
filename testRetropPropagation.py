""" Script de test de retropropagation : deux classes A (+1) et B (-1). On utilise le plan, et on se place
 sous les contraintes 0 <= x <= 10 et 0 <= y <= 10. Les classes A et B sont séparées par
 l'hyperplan d'équation y = 10-x ."""

from random import *
import retroGradientOpt as rgOpt
import matplotlib.pyplot as plt
import numpy as np
import listeOperation as lo

Nc = 5
u = [1,1]  #vecteur normal à l'hyperplan séparant les données
d = 2
c1 = 10**(-8)
c2 = 10**(-8)
mu_0 = 10**(-3)

print(abs(lo.ps(u,[5,5]))/(lo.ps(u,u)))

for k in range(1):

    " Generation des donnees "
    y = []
    Z = []
    nbExemples = 0
    nbRejetes = 0

    absA, absB = [], []
    ordA, ordB = [], []

    cardA = 0
    cardB = 0

    while (nbExemples < 500):
        a = randint(0,100)/10
        b = randint(0,100)/10
        dist = abs((lo.ps(u,[a,b]))/(lo.ps(u,u))-5)
        valide = (d==0) or ((d!=0) & (dist > 2))
        if (a > 10-b) & valide:
            Z.append([1,a,b])
            absA.append(a)
            ordA.append(b)
            y.append(1)
            nbExemples += 1
            cardA += 1
        elif (b < 10-b) & valide:
            Z.append([1,a,b])
            absB.append(a)
            ordB.append(b)
            y.append(-1)
            nbExemples += 1
            cardB += 1
        else:
            nbRejetes += 1

    plt.scatter(absA,ordA,s=10,c='r',marker='*')
    plt.scatter(absB,ordB,s=10,c='b',marker='o')
    plt.plot([0,10],[10,0],'orange')
    plt.title("Demi-distance à l'hyperplan : " + str(k/2) +"\n card(A) = "+str(cardA)+" ; card(B) = "+str(cardB))
    plt.show()

    print(3*np.eye(3,3))

    print("Session n° " + str(k+1))

    " Apprentissage sur les donnees generees "

    for j in range(2,Nc+1):
        print("          Apprentissage n°1")
        try:
            tE1 = rgOpt.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,j,c1,c2,mu_0,400,400,10,y,Z)
            abs1 = [(i+1) for i in range(len(tE1))]
            plt.plot(np.array(abs1),np.array(tE1),'red',label=str(tE1[len(tE1)-1]))
        except OverflowError:
            print("L'apprentissage a echoue -> erreur overflow")
        print("          Apprentissage n°2")
        try:
            tE2 = rgOpt.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,j,c1,c2,mu_0,400,400,10,y,Z)
            abs2 = [(i+1) for i in range(len(tE2))]
            plt.plot(np.array(abs2),np.array(tE2),'gold',label=str(tE2[len(tE2)-1]))
        except OverflowError:
            print("L'apprentissage a echoue -> erreur overflow")
        print("          Apprentissage n°3")
        try:
            tE3 = rgOpt.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,j,c1,c2,mu_0,400,400,10,y,Z)
            abs3 = [(i+1) for i in range(len(tE3))]
            plt.plot(np.array(abs3),np.array(tE3),'darkgreen',label=str(tE3[len(tE3)-1]))
        except OverflowError:
            print("L'apprentissage a echoue -> erreur overflow")
        print("          Apprentissage n°4")
        try:
            tE4 = rgOpt.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,j,c1,c2,mu_0,400,400,10,y,Z)
            abs4 = [(i+1) for i in range(len(tE4))]
            plt.plot(np.array(abs4),np.array(tE4),'cyan',label=str(tE4[len(tE4)-1]))
        except OverflowError:
            print("L'apprentissage a echoue -> erreur overflow")
        print("          Apprentissage n°5")
        try:
            tE5 = rgOpt.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,j,c1,c2,mu_0,400,400,10,y,Z)
            abs5 = [(i+1) for i in range(len(tE5))]
            plt.plot(np.array(abs5),np.array(tE5),'slategrey',label=str(tE5[len(tE5)-1]))
        except OverflowError:
            print("L'apprentissage a echoue -> erreur overflow")
        plt.legend()
        plt.title('Nc = ' + str(j) + 'neurones cachés.')
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("Erreur quadratique moyenne")
        plt.show()