#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as spr
from scipy.optimize import linprog
from numba import njit
import gurobipy as grb
import time


data_X = pd.read_csv("https://github.com/math-econ-code/mec_optim_2021-01/raw/master/data_mec_optim/marriage_personality-traits/Xvals.csv")
data_Y = pd.read_csv("https://github.com/math-econ-code/mec_optim_2021-01/raw/master/data_mec_optim/marriage_personality-traits/Yvals.csv")
data_affmat = pd.read_csv("https://github.com/math-econ-code/mec_optim_2021-01/raw/master/data_mec_optim/marriage_personality-traits/affinitymatrix.csv")


nobs = 1158

sdX = data_X.std().to_numpy()
sdY = data_Y.std().to_numpy()
mX = data_X.mean().to_numpy()
mY = data_Y.mean().to_numpy()

affmat = data_affmat.to_numpy()[0: 10, 1:]
Xvals = ((data_X-mX)/sdX).to_numpy()[0:nobs, :]
Yvals = ((data_Y-mY)/sdY).to_numpy()[0:nobs, :]
print('Xvals shape:', Xvals.shape)
print('Yvals shape:', Yvals.shape)
print('Affinity Matrix shape:', affmat.shape)



class random_panel:
    def __init__(self, init_matrix, time=20):
        self.time = time
        self.init_matrix = init_matrix
    def generate(self):
        X_panel = np.empty((self.time, self.init_matrix.shape[0], self.init_matrix.shape[1]))
        X_panel[0, :, :] = self.init_matrix
        for i in range(self.time - 1):
            X_panel[i+1, :, :] = X_panel[i, :, :] + np.random.randn(self.init_matrix.shape[0], self.init_matrix.shape[1])
        return X_panel



n = Xvals.shape[0]
t = 25


Xvals_dyna = random_panel(Xvals, t).generate()
Yvals_dyna = random_panel(Yvals.T, t).generate()

print(Xvals_dyna.shape)
print(Yvals_dyna.shape)

phi_dyna = Xvals_dyna @ affmat @ Yvals_dyna
print(phi_dyna.shape)
vecphi = phi_dyna.flatten()



p = np.ones((t, n, 1))/n
q = np.ones((t, n, 1))/n
d = np.concatenate((p,q), axis = None)

sparse_one = spr.csr_matrix(np.ones(n).reshape(1, n))

A = spr.kron(spr.identity(n*t), np.ones(n).reshape(1, n))
B = spr.kron(spr.kron(spr.identity(t), sparse_one), spr.identity(n))

Aconstr = spr.vstack([A, B])

#grb = Gurobi: Linear Programming Commercial Solver

m=grb.Model('Optimal Marriage')
x = m.addMVar(shape=t * n**2, name="x")
m.setObjective(vecphi @ x, grb.GRB.MAXIMIZE)
m.addConstr(Aconstr @ x == d, name="Constr")
m.optimize()

pi_panel = np.array(m.getAttr('x')).reshape(t, n, n)

def matching_of_man(man):
    for time in range(t):
        print('Period', time+1, ': Woman', np.argwhere(pi_panel[time, man-1,:] != 0)[0][0] + 1)


matching_of_man(1)
