
# coding: utf-8

import numpy as np
from scipy.special import binom
import pickle
import sys
import matplotlib.pyplot as plt

from load_data import load_data
from pareto import pareto_clean
from elicitation import EI, prefered_solution, OWA
from mixed_research import ILS


"""
A implémenter :
- Efficacité temporelle à p fixé
- Efficacité temporelle à n fixé
- Temps de calculs comparés avec et sans coupes pour p = 2 et n variable
- Taille des données pour PLS contre opti avec p fixé
- Taille des données pour PLS contre opti avec n fixé
- Valeur de la solution optimale pour les 2 algos et l'opti avec p fixé
- Valeur de la solution optimale pour les 2 algos et l'opti avec n fixé
"""

def load_results(n,p,to_load=range(20)):
    datas = load_data("dataSelection.txt")
    solution_directory = "solutions/n"+str(n)+"_p"+str(p)+"/"
    datas = datas[:n,:p]

    EI_sols = []
    ILS_sols = []

    EI_times = []
    ILS_times = []
    PLS_times = []
    PLSEI_times = []

    for iname in to_load:
        name = str(iname)
        solutions_file_name = solution_directory+"solutions_"+name
        time_file_name = solution_directory+"times_"+name

        with open(solutions_file_name, "rb") as f:
            solutions = pickle.load(f)
        best_solution, EI_solution, ILS_solution, weights = solutions

        with open(time_file_name, "rb") as f:
            times = pickle.load(f)
        t_PLS, t_EI, t_ILS = times

        EI_value = OWA(np.dot(EI_solution,datas),weights)
        #if p != 5 and p != 4:
        #    EI_value = OWA(np.dot(EI_solution,datas),weights)
        #else:
        #    EI_value = 0
        ILS_value = OWA(np.dot(ILS_solution,datas),weights)

        EI_sols.append(EI_value)
        ILS_sols.append(ILS_value)

        EI_times.append(t_EI)
        ILS_times.append(t_ILS)
        PLS_times.append(t_PLS)
        PLSEI_times.append(t_EI+t_PLS)

    return np.mean(EI_sols), np.mean(ILS_sols), np.mean(PLS_times), np.mean(EI_times), np.mean(PLSEI_times), np.mean(ILS_times)

def load_paretos(n,p,to_load=range(20)):
    datas = load_data("dataSelection.txt")
    solution_directory = "solutions/n"+str(n)+"_p"+str(p)+"/"
    datas = datas[:n,:p]

    paretos_len = []
    opts_len = []
    opts_sols = []

    for iname in to_load:
        name = str(iname)
        pls_file_name = solution_directory+"pls_"+name

        if p != 5 and p != 4:
            with open(pls_file_name, "rb") as f:
                paretos = pickle.load(f)
            paretos_len.append(len(paretos))
        else:
            paretos_len.append(0)

        opt_file_name = "optimum/PF_"+str(n)+"_"+str(p)+".txt"
        opt = []
        with open(opt_file_name, "r") as f:
            for l in f.readlines():
                opt.append(np.array([int(s) for s in l.split(" ")]))
            opt_len = len(opt)

        opts_len.append(opt_len)


        solutions_file_name = solution_directory+"solutions_"+name
        with open(solutions_file_name, "rb") as f:
            solutions = pickle.load(f)
        _,_,_,weights = solutions
        opt_sol = np.max(np.array([OWA(x,weights) for x in opt]))

        opts_sols.append(opt_sol)

    return np.mean(paretos_len), np.mean(opts_len), np.mean(opts_sols)

def time_efficiency_on_n(EI_mean_times,PLS_mean_times,PLSEI_mean_times,ILS_mean_times):
    x_axis = [10,20,30,40,50,60,80,100]
    plt.xlabel("n")
    plt.plot(x_axis,np.transpose([PLS_mean_times, EI_mean_times, PLSEI_mean_times, ILS_mean_times]))
    plt.title("Times for p = 2")
    plt.legend(["PLS","EI","PLS+EI","ILS"])
    plt.show()

def paretos_lenghts_on_n(paretos_len,opts_len):
    x_axis = [10,20,30,40,50,60,80,100]
    plt.xlabel("n")
    plt.plot(x_axis,paretos_len)
    plt.plot(x_axis,opts_len)
    plt.title("Count of solutions for p = 2")
    plt.legend(["PLS","Total count"])
    plt.show()

def value_quality_on_n(EI_sols,ILS_sols,Opt_sol):
    x_axis = [10,20,30,40,50,60,80,100]
    plt.xlabel("n")
    plt.plot(x_axis,EI_sols)
    plt.plot(x_axis,ILS_sols)
    plt.plot(x_axis,Opt_sol)
    plt.title("Quality of solutions for p = 2")
    plt.legend(["PLS-EI","ILS","Optimum"])
    plt.show()

if __name__ == "__main__":
    p = 2
    EI_sols = []
    ILS_sols = []
    EI_times = []
    PLS_times = []
    PLSEI_times = []
    ILS_times = []
    Pareto_len = []
    Opt_len = []
    Opt_sol = []
    for n, to_load in zip([10,20,30,40,50,60,80,100], [range(20)]*4+[range(5)]*2+[range(1)]*2):
        print((n,p))
        EI_sol, ILS_sol, PLS_time, EI_time, PLSEI_time, ILS_time = load_results(n,p,to_load=to_load)
        EI_sols.append(EI_sol)
        ILS_sols.append(ILS_sol)
        PLS_times.append(PLS_time)
        EI_times.append(EI_time)
        PLSEI_times.append(PLSEI_time)
        ILS_times.append(ILS_time)

        paretos_len, opts_len, opts_sols = load_paretos(n,p,to_load=to_load)
        Pareto_len.append(paretos_len)
        Opt_len.append(opts_len)
        Opt_sol.append(opts_sols)

    time_efficiency_on_n(EI_times,PLS_times,PLSEI_times,ILS_times)
    paretos_lenghts_on_n(Pareto_len,Opt_len)
    value_quality_on_n(EI_sols,ILS_sols,Opt_sol)
