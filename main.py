# coding: utf-8


import numpy as np
import pickle
import sys

from load_data import load_data
from pls import PLS

if __name__ == "__main__":

    datas = load_data("dataSelection.txt")
    n = 20
    k = n // 2

    subdatas = datas[:n]


    paretos = PLS(subdatas, n, k)

    name = sys.argv[1]
    f = open("solutions/pls_"+str(n)+"_"+name, 'wb')
    pickle.dump(paretos,f)
    f.close()
