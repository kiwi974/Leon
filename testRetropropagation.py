#Fichier de test des fonctions se trouvant dans le fichier retroGradient.py. Le but
#est ici de contrôler que tout se passe bien du point de vue des dimensions

import retroGradient as rg





"""Tests de la fonction initParam qui initialise les paramètres du réseau grâce
a une distribution gaussienne centree de variance 1/(n-1)"""
print(" ")
print(" ")
print(" ")
print("****************************************")
print("**************initParam*****************")
print("****************************************")
print(" ")

#Test pour 4 variables (+1 biais) et 1 neurone cache

W1_0,W2_0 = rg.initParam(4,1)
print("W1 vaut : " + str(W1_0))
print("W2 vaut : " + str(W2_0))

if ((len(W1_0)==1) & (len(W2_0) == 2) & (len(W1_0[0]) == 5)):
    print("Les resultats semblent coherents en terme de dimension")
else :
    print("Il y a un probleme dans les resultats concernant les dimensions")

print(" ")
print("****************************************")
print("****************************************")
print(" ")

#Test pour 7 variables (+1 biais) et 2 neurone cache

W1_1,W2_1 = rg.initParam(7,2)
print("W1 vaut : " + str(W1_1))
print("W2 vaut : " + str(W2_1))

if ((len(W1_1)==2) & (len(W2_1) == 3) & (len(W1_1[0])==8)):
    print("Les resultats semblent coherents en terme de dimension")
else :
    print("Il y a un probleme dans les resultats concernant les dimensions")





""" Test de la fonction calculant le modele """

print(" ")
print(" ")
print(" ")
print("****************************************")
print("*****************g**********************")
print("****************************************")

#test pour 4 variables (+1 biais) et 2 neurones cachés
ex0 = [1,2,5,8,7]
W1_ex0,W2_ex0 = rg.initParam(4,2)
g1 = rg.g(ex0,W1_ex0,W2_ex0)
print(g1)



""" Test de la fonction retro qui calcul les "delta" de la retropropagation du gradient """

print(" ")
print(" ")
print(" ")
print("****************************************")
print("*****************g**********************")
print("****************************************")

#Pour un reseau a 2 neuronnes caches, et 4 variables
ex1 = [1,2,6,7,3]
Nc1 = 2
y1 = 1
g1 = -1
potentiel1 = [1, -2.977356494704201, 2.739592564489479]
[o,W2_ex1] = rg.initParam(4,2)

print(rg.retro(Nc1,y1,g1,potentiel1,W2_ex1))





""" Test de la fonction retro qui efffectue la retropropagation du gradient """

print(" ")
print(" ")
print(" ")
print("****************************************")
print("*****************g**********************")
print("****************************************")
n2 = 10
Nc2 = 2
seuil2 = 10^(-2)
nbIterMax2 = 50

print(rg.retropropagation(n2,Nc2,seuil2,0.5,nbIterMax2,100,10))