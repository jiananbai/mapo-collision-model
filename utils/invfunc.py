import numpy as np
from scipy.optimize import minimize


class InverseFunc:
    def __init__(self, func):
        self.func = func

    def _squared_error(self, x, target):
        return (self.func(x) - target)**2

    def cal(self, target, lb=np.finfo(float).eps, ub=None):
        res = minimize(self._squared_error, x0=np.array(0.1), args=target, method='SLSQP', bounds=((lb, ub),), tol=1e-6)
        return res.x[0]
