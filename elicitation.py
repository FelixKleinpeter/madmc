# coding: utf-8

import numpy as np
import random as rd

import gurobipy as gp
from gurobipy import GRB

def DMweights(p):
    """Génère des poids pour le décideur et les range par ordre décroissant"""
    weights = np.array([rd.random() for _ in range(p)])
    weights = weights / sum(weights)
    return -np.sort(-weights)

def OWA(x,weights):
    sorted_x = -np.sort(-x)
    return np.dot(sorted_x,weights)

def prefer(x,y,weights,f=OWA):
    """Demande au décideur sa préférence entre deux solutions"""
    return f(x,weights) >= f(y,weights)

def prefered_solution(paretos, datas, weights):
    """Fonction de test qui permet de trouver la meilleure solution pour le
    décideur lorsque les poids représentant ses préférences sont connus"""
    images = np.dot(paretos,datas)
    owas = np.array([OWA(x,weights) for x in images])
    return paretos[np.argmax(owas)]

def EI(paretos, datas, weights, P_init = [], verbose=False):
    """ Algorithme d'élicitation icrémentale """
    P = P_init.copy()
    images = np.dot(paretos,datas)

    max_mmr = 0
    ratio = 1
    best_solution = paretos[0]

    # Tant que le ratio entre le MMR actuel et le MMR initial dépasse 0.1
    while ratio > 0.1:

        # Choisir une paire x,y selon le MMR
        xi, yi, m = MMR(images,P)
        x_image, y_image = images[xi], images[yi]
        x, y = paretos[xi], paretos[yi]


        # Initialisation du MMR maximum
        if max_mmr == 0:
            max_mmr = m
        # Actualiser le ratio de MMR
        ratio = m/max(max_mmr, 1)

        if verbose:
            print("Ratio de MMR actuel : "+str(ratio))

        # Requêtes au décideur et mise à jour de P et de la meilleure solution
        if prefer(x_image,y_image,weights):
            P.append((x_image,y_image))
            best_solution = x
        elif prefer(y_image,x_image,weights):
            P.append((y_image,x_image))
            best_solution = y

    return best_solution, P


def PMR(x,y,P):
    """ Pairwise Max Regret pour une fonctio OWA avec Gurobi """
    p = len(x)

    m = gp.Model("PMR")

    # Supprime les affichages des détails des calculs
    m.setParam( 'OutputFlag', False )

    # Variables
    w = []
    for i in range(p):
        w.append(m.addVar(vtype=GRB.CONTINUOUS, name="w_"+str(i)))

    # Objectif
    s_x = -np.sort(-x)
    s_y = -np.sort(-y)
    m.setObjective(sum(wi*yi - wi*xi for wi,xi,yi in zip(w,s_x,s_y)), GRB.MAXIMIZE)

    # Contraintes
    for i in range(p):
        m.addConstr(w[i] >= 0, "w_"+str(i)+" positivity")
    m.addConstr(sum(wi for wi in w) <= 1, "w_i sum <= 1")
    m.addConstr(sum(wi for wi in w) >= 1, "w_i sum >= 1")
    for i in range(p-1):
        m.addConstr(w[i] - w[i+1] >= 0, "w_i >= wi+1 "+str(i))
    for i, (a,b) in enumerate(P):
        s_a = -np.sort(-a)
        s_b = -np.sort(-b)
        m.addConstr(sum(wi*ai-wi*bi for wi,ai,bi in zip(w,s_a,s_b)) >= 0,
            "ab_"+str(i))

    m.optimize()

    return np.array([v.x for v in m.getVars()]), m.objVal

def MMR(X,P,verbose=False):
    """ Min Max Regret étant donné P """
    # Initialisation des variables
    xi = 0
    yi = 0
    U = 1e8
    if verbose:
        print("Nombre de solution explorées dans le MMR : ")
    for i, z in enumerate(X):
        yi_, v = MR(z,X,P,U)
        if v < U:
            xi = i
            U = v
            yi = yi_
        if i%100==0 and verbose:
            print(i)
    return xi, yi, U

def MR(x,X,P,U=1e8):
    """ Max Regret de x étant donné P avec options de coupes """
    # Initialisation des variables
    L = -1e8
    yi = 0
    for i, z in enumerate(X):
        _, v = PMR(x,z,P)
        if v > L:
            yi = i
            L = v
            # Coupe alpha-beta
            if L >= U:
                return i, 1e8
    return yi, L
