from retroGradientOpt import *
import math

#### PARAMETRES DU RESEAU DE TEST ####
n = 2
Nc = 2

N = (n+1)*Nc + Nc+1

#### ENTREE ####
x = [1,1,2]



""" FONCTION : initParam
test : les données étant générées aléatoirement, on vérifira simplement que le vecteur de retour 
est de la bonne dimension."""

print(' ')
print('---------- TEST initParam ----------')
print('Test n°1 : N = len(initParam(n,Nc) : ' +str(N==len(initParam(n,Nc))))
print(' ')





""" FONCTION : g """
print(' ')
print('---------- TEST g ----------')
Wg = [1,2,3,4,3,2,1,1,1]
modele, potentiel = g(x,n,Nc,Wg)
print('Test n°1 : potentiel = [1,9,11,21] : ' + str(potentiel == [1,9,11,21]))
print('Test n°2 : modele = tanh(11)+tanh(9)+1 : ' + str((modele==(tanh(11)+tanh(9)+1))))
print(' ')





""" FONCTION : EQM """
print(' ')
print('---------- TEST EQM ----------')
y = [1,1,-1,1]
g = [0.98,0.74,-0.87,0.53]
err, errk = EQM(y,g)
errk[0] = ((errk[0]*100)//1)/100  #### car Python renvoie 1-0.98 = 0.02000000000018
errkB = (errk[0]==0.02)&(errk[1]==0.26)&(errk[2]==-0.13)&(errk[3]==0.47)
print('Test n°1 : errk = [0.02,0.26,-0.13,0.53] : ' + str(errkB))
print('Test n°2 : err = 0.5*sqrt(0.02**2+0.26**2+0.13**2+0.53**2) : ' + str(err-(0.5*sqrt(0.02**2+0.26**2+0.13**2+0.53**2)) <= 10**(-8)))
print(' ')





""" FONCTION : retro """
print(' ')
print('---------- TEST : retro ----------')
y = 1.0
g = 0.95
delta = retro(n,Nc,y,g,potentiel,Wg)
deltaPrevu = [21/(cosh(1)**2),21/(cosh(9)**2),21/(cosh(11)**2),(-2)*(y-g)/(cosh(21)**2)]
deltaB = (delta[0]==deltaPrevu[0])&(delta[1]==deltaPrevu[1])&(delta[2]==deltaPrevu[2])&(delta[3]==deltaPrevu[3])
print('Test n°1 : delta = deltaPrevu : ' + str(deltaB))
print(' ')




""" FONCTION : gradient """
print(' ')
print('---------- TEST : gradient ----------')
nbExemples = 2
y = [1,-1]
sortiesEx = [0.74,-0.87]
Z = [[1,0,0],[1,10,10]]
potentielsEx = [[1,2,3,5],[1,2,4,2]]
dw, dg = gradient(nbExemples,n,Nc,y,sortiesEx,potentielsEx,Z,Wg)