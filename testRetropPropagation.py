""" Script de test de retropropagation : deux classes A (+1) et B (-1). On utilise le plan, et on se place
 sous les contraintes 0 <= x <= 10 et 0 <= y <= 10. Les classes A et B sont séparées par
 l'hyperplan d'équation y = 10-x ."""

from random import *
import retroGradient as rg
import retroGradientOpt as rgOpt
import matplotlib.pyplot as plt
import numpy as np

Nc = 12

for i in range(10):

    " Generation des donnees "
    y = []
    Z = []
    nbRejetes = 0

    absA, absB = [], []
    ordA, ordB = [], []

    for k in range(200):
        a = randint(0,10)*random()
        b = randint(0,10)*random()
        if (a > 10-b):
            Z.append([1,a,b])
            absA.append(a)
            ordA.append(b)
            y.append(1)
        elif (b < 10-b):
            Z.append([1,a,b])
            absB.append(a)
            ordB.append(b)
            y.append(-1)
        else:
            nbRejetes += 1

    #plt.plot(absA,ordA,'r')
    #plt.plot(absB,ordB,'b')
    #plt.show()

    print("Session n° " + str(i+1))

    " Apprentissage sur les donnees generees "

    for j in range(1):
        print("          Apprentissage n°1")
        tE1 = rgOpt.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,2,0.1,10**(-2),10**(-1),400,400,10,y,Z)
        print("          Apprentissage n°2")
        tE2 = rgOpt.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,2,0.1,10**(-2),10**(-1),400,400,10,y,Z)
        print("          Apprentissage n°3")
        tE3 = rgOpt.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,2,0.1,10**(-2),10**(-1),400,400,10,y,Z)
        print("          Apprentissage n°4")
        tE4 = rgOpt.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,2,0.1,10**(-2),10**(-1),400,400,10,y,Z)
        print("          Apprentissage n°5")
        tE5 = rg.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,2,0.1,10**(-2),10**(-1),400,400,10,y,Z)
        abs1 = [(i+1) for i in range(len(tE1))]
        abs2 = [(i+1) for i in range(len(tE2))]
        abs3 = [(i+1) for i in range(len(tE3))]
        abs4 = [(i+1) for i in range(len(tE4))]
        abs5 = [(i+1) for i in range(len(tE5))]
        plt.plot(np.array(abs1),np.array(tE1),'red')
        plt.plot(np.array(abs2),np.array(tE2),'gold')
        plt.plot(np.array(abs3),np.array(tE3),'darkgreen')
        plt.plot(np.array(abs4),np.array(tE4),'cyan')
        plt.plot(np.array(abs5),np.array(tE5),'slategrey')
        plt.title('Nc = ' + str(i) + 'neurones cachés.')
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("Erreur quadratique moyenne")
        plt.show()