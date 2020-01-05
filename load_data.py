# coding: utf-8

import numpy as np

def load_data(file_name):
    datas = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            datas.append([int(s) for s in line.lstrip().rstrip().split(" ")])

    return np.array(datas)
