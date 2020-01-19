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

    for (n,p) in [(20,3)]:
        print((n,p))
        for iname in range(20):
            name = str(iname)
            print(name)


            datas = load_data("dataSelection.txt")
            k = n // 2
            subdatas = datas[:n,:p]
            solution_directory = "../MADMC_Solutions/solutions/n"+str(n)+"_p"+str(p)+"/"
            if not os.path.exists(solution_directory):
                os.mkdir(solution_directory)

            # Pareto locale search
            print("==== PLS ====")
            t0 = time.time()
            paretos = PLS(subdatas, n, k, verbose = VERBOSE)
            t_PLS = time.time() - t0

            with open(solution_directory+"pls_"+name, 'wb') as f:
                pickle.dump(paretos,f)

            weights = DMweights(p)
            best_solution_on_paretos = prefered_solution(paretos, subdatas, weights)

            print("==== EI ====")
            # Elicitation incrémentale
            t0 = time.time()
            best_solution_EI, _ = EI(paretos, subdatas, weights, verbose=VERBOSE)
            t_EI = time.time() - t0

            print("==== ILS ====")
            #  Interactive Local Search
            t0 = time.time()
            best_solution_ILS = ILS(subdatas, n, k, weights, verbose=VERBOSE)
            t_ILS = time.time() - t0

            with open(solution_directory+"solutions_"+name, 'wb') as f:
                pickle.dump([best_solution_on_paretos, best_solution_EI, best_solution_ILS, weights],f)

            with open(solution_directory+"times_"+name, 'wb') as f:
                pickle.dump([t_PLS, t_EI, t_ILS],f)
