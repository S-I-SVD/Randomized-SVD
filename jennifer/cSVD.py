#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:17:08 2020

@author: jenzyy
"""
import imageio 
import matplotlib.pyplot as plt
import numpy as np


def cSVD(X,k,q,p):
    l = k+p
    m = X.shape[0]
    n = X.shape[1]
    psi = np.random.rand(l,m)
    Y = psi @ X
    B = Y @ Y.T
    B = (1/2) * (B + B.T)
    D,T = np.linalg.eig(B)
    T = T[0:k].T
    D = D[0:k]
    D = np.diag(D)
    S1 = np.sqrt(D)
    V1 = Y.T @ T @ np.linalg.inv(S1)
    U1 = X @ V1
    U, S, QT = np.linalg.svd(U1)
    Q = QT.T
    V = V1 @ Q
    S = np.diag(S)
    
    approx = U[:,:k] @ S @ VT
    return approx