""" Script de test de retropropagation : deux classes A (+1) et B (-1). On utilise le plan, et on se place
 sous les contraintes 0 <= x <= 10 et 0 <= y <= 10. Les classes A et B sont séparées par
 l'hyperplan d'équation y = 10-x ."""

from random import *
import retroGradient as rg
import matplotlib.pyplot as plt

" Generation des donnees "

y = []
Z = []
nbRejetes = 0

absA, absB = [], []
ordA, ordB = [], []

for i in range(200):
    a = randint(0,10)*random()
    b = randint(0,10)*random()
    if (a > 10-b):
        Z.append([a,b])
        absA.append(a)
        ordA.append(b)
        y.append(1)
    elif (b < 10-b):
        Z.append([a,b])
        absB.append(a)
        ordB.append(b)
        y.append(-1)
    else:
        nbRejetes += 1

plt.plot(absA,ordA,'r')
plt.plot(absB,ordB,'b')

plt.show()


" Apprentissage sur les donnees generees "

rg.retropropagation("/home/ray974/Learning/Data/bdd_dev.db",2,3,1.5,2,50,100,10,y,Z)