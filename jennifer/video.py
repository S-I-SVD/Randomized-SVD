# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:26:10 2020

@author: jenni
"""

import matplotlib.pyplot as plt
import numpy as np
import skvideo as vid
import skvideo.io
import skvideo.utils

'''
# set wd
import os
os. chdir('./Documents/GitHub/Randomized-SVD/jennifer')
'''

# svd algorithm
def rand_svd(X,r,q,p):
    ny = X.shape[1]
    P =  np.random.standard_normal(size=(ny, r+p))
    Z = X @ P
    for k in range(1,q):
        Z = X @ (X.conj().transpose() @ Z)
    Q,R = np.linalg.qr(Z,mode='reduced')
    Y = Q.conj().transpose() @ X
    Uy , S , V = np.linalg.svd(Y, full_matrices=False)
    U = Q @ Uy
    approx = U @ np.diag(S)[:, :r] @ V[:r, :]
    return approx

# Load video
video = vid.io.vread('school_full.mp4')

'''
# color separated
# separate colors
vshape = video.shape
red = video[:,:,:,0]
green = video[:,:,:,1]
blue = video[:,:,:,2]

# reshape
num = vshape[0]
red_flat = red.reshape(num,-1)
green_flat = green.reshape(num,-1)
blue_flat = blue.reshape(num,-1)

# apply SVD
r = 5
q = 1
p = 3
approx = np.empty(shape=(red_flat.shape[0], red_flat.shape[1],3))
approx[:,:,0]=rand_svd(red_flat, r, q, p)
approx[:,:,1]=rand_svd(green_flat, r, q, p)
approx[:,:,2]=rand_svd(blue_flat, r, q, p)
'''

# reshape
vshape = video.shape
num = vshape[0]
flat = video.reshape(num,-1)

# apply SVD
r = 2 # rank
q = 1 # power iterations 
p = 3 # oversampling parameter
approx = rand_svd(flat, r, q, p)


# reconstruct video
vapprox = approx.reshape(vshape)

#output
vid.io.vwrite('school_full_rank_2.mp4', vapprox)
