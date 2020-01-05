# coding: utf-8

import numpy as np
import random as rd

from sortedcontainers import SortedSet
from pareto import pareto_clean, neighbours, binary


def PLS(datas,n,k):
    solutions = []
    all_solutions = SortedSet()
    solution = np.zeros((n), dtype=int)
    solution[rd.sample(range(n),k)] = 1

    # Les solutions pareto optimales
    solutions = np.array(solution)

    # Tous les binaires les solutions visitées
    all_solutions_binaries.add(binary(solution))

    solutions_to_explore = [solution]

    while len(solutions_to_explore) > 0:
        # Choix d'une solution à explorer
        new_solution = solutions_to_explore.pop()

        # Calcul de ses voisins
        all_neighbours = neighbours(new_solution)

        for neighbour in all_neighbours:
            if not binary(neighbour) in all_solutions_binaries:
                solutions_to_explore.append(neighbour)
                all_solutions_binaries.add(binary(neighbour))

        # On supprime les solutions non optimales selon Pareto
        solutions = pareto_clean(all_neighbours + solutions,datas)

    return solutions
