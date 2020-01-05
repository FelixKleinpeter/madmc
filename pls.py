# coding: utf-8

import numpy as np
import random as rd

from sortedcontainers import SortedSet
from pareto import pareto_insert, pareto_clean, neighbours, binary


def PLS(datas,n,k):

    solution = np.zeros((n), dtype=int)
    solution[rd.sample(range(n),k)] = 1

    # Les solutions pareto optimales
    solutions = [solution]

    # Toutes les solutions visitées en base 10
    solutions_binaries = SortedSet([binary(solution)])

    # Solutions à explorer
    solutions_to_explore = [solution]

    while len(solutions_to_explore) > 0:
        # Choix d'une solution à explorer
        new_solution = solutions_to_explore.pop()

        # Calcul de ses voisins
        all_neighbours = neighbours(new_solution)

        # Ajout des voisins dans les solutions à explorer
        for neighbour in all_neighbours:
            if not binary(neighbour) in solutions_binaries:
                solutions_binaries.add(binary(neighbour))
                if pareto_insert(solutions, neighbour, datas):
                    solutions_to_explore.append(neighbour)


        if len(solutions_to_explore) % 10 == 0:
            print(len(solutions_to_explore))

    return solutions
