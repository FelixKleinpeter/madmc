# coding: utf-8

import numpy as np

def pareto_dominates(s1,s2):
    """ Domination de Pareto au sens large """
    return np.array([s1[i] >= s2[i] for i in range(len(s1))]).all()

def binary(s):
    """ Retourne la valeure en base 10 d'une solution binaire """
    return sum(p * 2**i for i,p in enumerate(s))

def neighbours(solution):
    """ Trouve les voisins d'une solution en lui appliquant des permutations 1 - 1 """
    selected = np.where(solution == 1)[0]
    unselected = np.where(solution == 0)[0]
    solutions = []
    for i in selected:
        for j in unselected:
            neighbour = solution.copy()
            neighbour[i] = 0
            neighbour[j] = 1
            solutions.append(neighbour)
    return solutions


def pareto_clean(solutions, datas):
    """ Supprime toutes les solutions Pareto dominées d'un ensemble """
    solutions_images = np.dot(solutions,datas)
    pareto_optima = []
    for i, s1 in enumerate(solutions):
        add = True
        for j, s2 in enumerate(pareto_optima):
            if pareto_dominates(solutions_images[j],solutions_images[i]):
                add = False
                break
        if add:
            pareto_optima.append(s1)
    return pareto_optima
