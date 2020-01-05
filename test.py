# coding: utf-8

import numpy as np
import pickle
import sys

from load_data import load_data
from pareto import pareto_clean, binary


if __name__ == "__main__":
    file_name = sys.argv[1]

    with open(file_name, "rb") as f:
        solutions = np.array(pickle.load(f))
    n = solutions.shape[1]

    datas = load_data("dataSelection.txt")
    subdatas = datas[:n]

    images = np.dot(solutions,subdatas)
    cleaned_solutions = pareto_clean(solutions,subdatas)

    bin = [binary(s) for s in solutions]
    cleaned_bin = [binary(s) for s in cleaned_solutions]

    print(len(solutions))
    print(len(cleaned_solutions))

    assert len(solutions) == len(cleaned_solutions)
