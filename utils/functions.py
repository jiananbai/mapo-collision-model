import numpy as np

def fairness_func(x, param=3):
    return (np.exp(param * x) - 1) / (np.exp(param) - 1)
