#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 23:22:10 2020

@author: katie
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from timeit import timeit
from mpl_toolkits import mplot3d
import scipy
from fractions import Fraction
import svd as svd
from matplotlib.image import imread
from skimage import color
from skimage import io

def singularvaluesize(M):
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    U, S, VT = np.linalg.svd(M_stacked, full_matrices=False)
    S = np.diag(S) 
    return S

def reshapeW(M,W):
    M_type = M.dtype
    r, c = M.shape[:2]
    #reshaping back into original matrix
    W = W.reshape(r, c, -1)
    #dimension issues
    if W.shape[2] == 1:
        W.shape = W.shape[:2] #should output a two dimensional matrix
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        W = np.clip(W, 0, 255)
    else:
        W = np.clip(W, 0, 1)
    return W.astype(M_type)

def padimage(M,sizematrix): 
    M_padded = np.zeros(sizematrix.shape)
    M_padded[:M.shape[0],:M.shape[1]] = M
    return M_padded

def reversepad(W,originalwatermark):
    sizes = originalwatermark.shape
    W = W[:sizes[0],:sizes[1]]
    return W

def lowrankapprox(M_stacked, k): 
    U, S, VT = np.linalg.svd(M_stacked, full_matrices=False)
    S = np.diag(S)
    M_approx = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
    return M_approx
