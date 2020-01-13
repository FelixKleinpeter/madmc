# coding: utf-8


import numpy as np
import pickle
import sys
import time
import os

from load_data import load_data
from pls import PLS
from elicitation import EI, DMweights, prefered_solution
from mixed_research import ILS

VERBOSE = True

if __name__ == "__main__":

    """
    datas = load_data("dataSelection.txt")
    n = 30
    k = n // 2
    p = 5 # A lancer
    subdatas = datas[:n,:p]

    # À développer sur PLS et EI : 50,2 60,2 50,3
    # Sur plusieurs instances : 40,4 60,3 80,2 100,2 200,2

    # Calculer les coefficients binomiaux pour chaque valeur (approx du nb de solutions possibles)

    solution_directory = "solutions/n"+str(n)+"_p"+str(p)+"/"
    if not os.path.exists(solution_directory):
        os.mkdir(solution_directory)
    """
    for (n,p) in [(10,2), (10,3), (10,4), (10,5), (10,6)]:
    #for (n,p) in [(100,2)]:
        print((n,p))
        for iname in range(20):
            #, (80,2), (100,2)]):
            name = str(iname)
            print(name)


            datas = load_data("dataSelection.txt")
            k = n // 2
            subdatas = datas[:n,:p]
            solution_directory = "solutions/n"+str(n)+"_p"+str(p)+"_special/"
            prev_solution_directory = "solutions/n"+str(n)+"_p"+str(p)+"/"
            if not os.path.exists(solution_directory):
                os.mkdir(solution_directory)

            # Pareto locale search
            t0 = time.time()
            #paretos = PLS(subdatas, n, k, verbose = VERBOSE)
            with open(prev_solution_directory+"pls_"+name, 'rb') as f:
                paretos = pickle.load(f)
            t_PLS = time.time() - t0

            #with open(solution_directory+"pls_"+name, 'wb') as f:
            #    pickle.dump(paretos,f)

            weights = DMweights(p)
            best_solution_on_paretos = prefered_solution(paretos, subdatas, weights)

            # Elicitation incrémentale
            t0 = time.time()
            best_solution_EI, _ = EI(paretos, subdatas, weights, verbose=VERBOSE)
            t_EI = time.time() - t0

            #  Interactive Local Search
            t0 = time.time()
            #best_solution_ILS = ILS(subdatas, n, k, weights, verbose=VERBOSE)
            best_solution_ILS = 0
            t_ILS = time.time() - t0

            with open(solution_directory+"solutions_"+name, 'wb') as f:
                pickle.dump([best_solution_on_paretos, best_solution_EI, best_solution_ILS, weights],f)
                #pickle.dump([[], [], best_solution_ILS, weights],f)

            with open(solution_directory+"times_"+name, 'wb') as f:
                pickle.dump([t_PLS, t_EI, t_ILS],f)
