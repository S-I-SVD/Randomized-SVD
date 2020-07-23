#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:56:07 2020

@author: jenzyy
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np

'''
# set wd
import os
os. chdir('./Documents/GitHub/Randomized-SVD/jennifer/watermark')
'''

# read in image
im = imageio.imread("rose.jpg")

# watermark
W = imageio.imread("cat.jpg")
#W = np.random.rand(rows,rows)
Wp = imageio.imread("dog.jpg")
# scale
a = 0.1

def im_stack(im):
    im_type = im.dtype
    im = im.astype(np.float64)
    rows,cols = im.shape[:2]
    im_stacked = im.reshape(rows,-1)
    return im_stacked, im_type

def im_stack_s(im, im_type):
    rows = im.shape[0]
    cols = im.shape[1]//3
    im_m = im.reshape(rows, cols, -1)
    im_m = im_m.astype(im_type)
    plt.imshow(im_m)
    plt.show()
    
im_stacked, im_type = im_stack(im)
W_stacked, W_type = im_stack(W)
Wp_stacked, Wp_type = im_stack(Wp)

def watermark_image(im, W, a):
    rows,cols = im.shape[:2]
    U,S,V = np.linalg.svd(im,full_matrices = False)
    Wp = np.pad(W,[(0, rows - W.shape[0]), (0, rows - W.shape[1])])
    Aw = np.diag(S)+a*Wp
    Uw,Sw,Vw = np.linalg.svd(Aw,full_matrices = True)
    marked = U @ np.diag(Sw) @ V
    return marked, Uw, S, Vw

# show output
marked, Uw, S, Vw = watermark_image(im_stacked, W_stacked,0.1)
im_stack_s(marked,im_type)

# extract watermark
def watermark_extract(marked, Uw, S,Vw, a):
    Um, Sm, Vm = np.linalg.svd(marked)
    M = (Uw @ np.diag(Sm) @ Vw - np.diag(S))/a
    #rows = len(S)
    #Mp = np.pad(M,[(0, M.shape[0]- rows), (0, M.shape[1] - rows)])
    return M

# test output
M = watermark_extract(marked, Uw, S, Vw, a)
Mrow, Mcol = W_stacked.shape
M = M[:Mrow, :Mcol]
im_stack_s(M, W_type)

# extract wrong watermark
marked_p, Uw_p, S_p, Vw_p = watermark_image(im_stacked, Wp_stacked,0.1)
Mp = watermark_extract(marked, Uw_p, S, Vw_p, a)
Mprow, Mpcol = Wp_stacked.shape
Mp = Mp[:Mprow, :Mpcol]
im_stack_s(Mp, Wp_type)