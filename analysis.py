
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

def load_results(n,p,to_load=range(20),special=False):
    datas = load_data("dataSelection.txt")
    solution_directory = "../MADMC_Solutions/solutions/n"+str(n)+"_p"+str(p)+"/"
    special_solution_directory = "../MADMC_Solutions/solutions/n"+str(n)+"_p"+str(p)+"_special/"
    datas = datas[:n,:p]

    EI_sols = []
    ILS_sols = []


    EI_times = []
    ILS_times = []
    PLS_times = []
    PLSEI_times = []
    Special_times = []

    for iname in to_load:
        name = str(iname)
        solutions_file_name = solution_directory+"solutions_"+name
        time_file_name = solution_directory+"times_"+name
        if special:
            special_time_file_name = special_solution_directory+"times_"+name
            with open(special_time_file_name, "rb") as f:
                times = pickle.load(f)
            _, t_EI_spe, _ = times
            Special_times.append(t_EI_spe)

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

    return np.mean(EI_sols), np.mean(ILS_sols), np.mean(PLS_times), np.mean(EI_times), np.mean(PLSEI_times), np.mean(ILS_times), np.mean(Special_times)

def load_paretos(n,p,to_load=range(20)):
    datas = load_data("dataSelection.txt")
    solution_directory = "../MADMC_Solutions/solutions/n"+str(n)+"_p"+str(p)+"/"
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

        opt_file_name = "../MADMC_Solutions/optimum/PF_"+str(n)+"_"+str(p)+".txt"
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

def time_efficiency_on_n(EI_mean_times,PLS_mean_times,PLSEI_mean_times,ILS_mean_times,x_axis):
    plt.xlabel("n")
    plt.plot(x_axis,np.transpose([PLS_mean_times, EI_mean_times, PLSEI_mean_times, ILS_mean_times]))
    plt.title("Times for p = 3")
    plt.legend(["PLS","EI","PLS+EI","ILS"])
    plt.show()

def paretos_lenghts_on_n(paretos_len,opts_len,x_axis):
    plt.xlabel("n")
    plt.plot(x_axis,paretos_len)
    plt.plot(x_axis,opts_len)
    plt.title("Count of solutions for p = 3")
    plt.legend(["PLS","Total count"])
    plt.show()

def value_quality_on_n(EI_sols,ILS_sols,Opt_sol,x_axis):
    plt.xlabel("n")
    plt.plot(x_axis,EI_sols)
    plt.plot(x_axis,ILS_sols)
    plt.plot(x_axis,Opt_sol)
    plt.title("Quality of solutions for p = 3")
    plt.legend(["PLS-EI","ILS","Optimum"])
    plt.show()

def time_efficiency_special(EI_times, Special_times,x_axis):
    plt.xlabel("p")
    plt.plot(x_axis,EI_times)
    plt.plot(x_axis,Special_times)
    plt.title("Speed comparison for n = 10")
    plt.legend(["With cuts","Without cuts"])
    plt.show()

if __name__ == "__main__":
    n = 10
    EI_sols = []
    ILS_sols = []
    EI_times = []
    PLS_times = []
    PLSEI_times = []
    ILS_times = []
    Special_times = []
    Pareto_len = []
    Opt_len = []
    Opt_sol = []

    x_axis = [2,3,4,5,6]
    for p, to_load in zip(x_axis, [range(20)]*5):
        print((n,p))
        EI_sol, ILS_sol, PLS_time, EI_time, PLSEI_time, ILS_time, Special_time = load_results(n,p,to_load=to_load,special=True)
        EI_sols.append(EI_sol)
        ILS_sols.append(ILS_sol)
        PLS_times.append(PLS_time)
        EI_times.append(EI_time)
        PLSEI_times.append(PLSEI_time)
        ILS_times.append(ILS_time)
        Special_times.append(Special_time)

        paretos_len, opts_len, opts_sols = load_paretos(n,p,to_load=to_load)
        Pareto_len.append(paretos_len)
        Opt_len.append(opts_len)
        Opt_sol.append(opts_sols)



    time_efficiency_special(EI_times, Special_times,x_axis)
    #time_efficiency_on_n(EI_times,PLS_times,PLSEI_times,ILS_times,x_axis)
    #paretos_lenghts_on_n(Pareto_len,Opt_len,x_axis)
    #value_quality_on_n(EI_sols,ILS_sols,Opt_sol,x_axis)
