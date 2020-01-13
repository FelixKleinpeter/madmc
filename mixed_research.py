# coding: utf-8

import numpy as np
import random as rd

import gurobipy as gp
from gurobipy import GRB

from pareto import pareto_insert, pareto_clean, neighbours, binary
from elicitation import EI, OWA, prefered_solution

def add_constraint_set(P,new_P):
    """ Ajoute un ensemble de nouveaux couples à P si P ne les contient pas déjà """
    final_P = P.copy()
    for (a,b) in new_P:
        add = True
        for (c,d) in P:
            if (a == c).all() and (b == d).all():
                add = False
        if add:
            final_P.append((a,b))
    return final_P

def ILS(datas,n,k,weights,verbose=False):
    """ Interactive Local Search """

    # Initialisation de la solution aléatoire
    solution = np.zeros((n), dtype=int)
    solution[rd.sample(range(n),k)] = 1
    P = []

    while 1:
        # Calcul des voisins
        all_neighbours = neighbours(solution)
        all_neighbours.append(solution)

        # Selection des voisins Pareto optimaux
        paretos = pareto_clean(all_neighbours, datas)

        # Choix de la meilleure solution par élicitation incrémentale
        best_solution, new_P = EI(paretos, datas, weights, P_init=P, verbose=False)

        # Mise à jour de l'ensemble des contraintes
        P = add_constraint_set(P,new_P)

        # Arrêt lorsqu'un optimum local est trouvé
        if (best_solution == solution).all():
            return best_solution

        # Mise à jour de la solution optimale
        solution = best_solution

        if verbose:
            print("Valeur de la solution optimale : "+str(OWA(np.dot(solution,datas),weights)))
