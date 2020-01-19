# coding: utf-8

import numpy as np
import pickle
import sys

from load_data import load_data
from pareto import pareto_clean
from elicitation import EI, prefered_solution, OWA
from mixed_research import ILS

def show_solution(name,s,datas,weights):
    print(name+"\t : "+str(s)+", Image : "+str(np.dot(s,datas))+", Valeur : "+str(OWA(np.dot(s,datas),weights)))


if __name__ == "__main__":
    n = 20
    p = 3
    name = "1"

    paretos_file_name = "../MADMC_Solutions/solutions/pls_n"+str(n)+"_p"+str(p)+"_"+name
    solutions_file_name = "../MADMC_Solutions/solutions/solutions_n"+str(n)+"_p"+str(p)+"_"+name

    with open(paretos_file_name, "rb") as f:
        paretos = np.array(pickle.load(f))

    datas = load_data("dataSelection.txt")
    subdatas = datas[:n,:p]

    print("========= Variables =========")
    print("n = "+str(n))
    print("p = "+str(p))

    print("========= Vérification de la Pareto-optimalité des solutions =========")
    images = np.dot(paretos,subdatas)
    cleaned_paretos = pareto_clean(paretos,subdatas)

    print("Nombre de solutions retournées par l'algorithme : "+str(len(paretos)))
    print("Nombre de solutions après un filtre de Pareto : "+str(len(cleaned_paretos)))

    assert len(paretos) == len(cleaned_paretos)

    print("========= Analyse des solutions optimales =========")
    with open(solutions_file_name, "rb") as f:
        solutions = pickle.load(f)

    best_solution, EI_solution, ILS_solution, weights = solutions

    show_solution("Best",best_solution,subdatas,weights)
    show_solution("EI",EI_solution,subdatas,weights)
    show_solution("ILS",ILS_solution,subdatas,weights)

    for a,b in zip(best_solution, EI_solution):
        assert a == b
