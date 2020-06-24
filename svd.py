#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:31:25 2020

@author: katie
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from timeit import timeit


img = imageio.imread('imageio:chelsea.png')

#Function for doing a regular SVD
def regularSVD(M):
    #Create economy SVD for M
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    #Turn singular values into a diagonal matrix
    #Return approximate image
    return U, S, VT

def regSVDapprox(M, r):
    U, S, VT = regularSVD(M)
    S = np.diag(S)
    # Construct approximate image from U, S, VT with rank k
    M_approx = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    return M_approx

def rSVDapprox(M,r,poweriterations,oversample):
    U, S, VT = rSVD(M,r,poweriterations,oversample)
    M_approx = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    return M_approx
    
#Function for doing a color image channel-by-channel
def colorchannelsvd(M,k): #M is the matrix, k is the rank of the approximation
    approx = np.empty_like(M)
    Red = M[:,:,0]
    Green = M[:,:,1]
    Blue = M[:,:,2]
    colors = (Red, Green, Blue)
    for i in range(3): #range(3) since there's three color layers
        approx[:,:,i] = regularsvd(colors[i], k)
    return approx  #returns the approximation of the original color image M

def colorstackingsvd(M,k): #M is the matrix, k is the rank of the approximation
    #regularSVD
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    M_rank = np.linalg.matrix_rank(M_stacked)
    M_approx_stacked = regularsvd(M_stacked, k)
    M_approx = M_approx_stacked.reshape(r, c, -1)
    return M_approx



def singularvalueplot(M): #plotting the singular values
    #only use on 1 layer matrix (stacked)
    U, S, VT = regularSVD(M)
    plt.semilogy(S)
    plt.title('Singular Values')
    plt.show()
    
def cumsumplot(M): #how much of the original data is captured
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
    plt.title('Singular Values: Cumulative Sum')
    plt.show()
    
def rSVD(M,r,poweriterations,oversample):
    #Sample column space of M with P matrix
    M = imagestack(M) #imagestack turns it into a 2D matrix
    r, c = M.shape
    #random projection matrix
    P = np.random.rand(c,r+oversample)
    #sampling from M's column space
    Z = M @ P
    #power iterations
    for k in range(poweriterations):
        Z = M @ (M.T @ Z)
    #QR factorization
    Q, R = np.linalg.qr(Z,mode='reduced')
    #Compute SVD on projected Y = Q.T @ X
    Y = Q.T @ M
    UY, S, VT = np.linalg.svd(Y,full_matrices=0)
    U = Q @ UY
    S = np.diag(S)
    return U, S, VT

def imagestack(M):#turns every input image (color or not) into a 2-dimensional matrix
    M_type = M.dtype
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    return M_stacked

def imagerestack(original, M):
    r, c = original.shape[:2]
    M_approx = M.reshape(r, c, -1)
    return M_approx

#def svdtime():
    #print("Regular SVD, rank {0}: {1} seconds".format(k, timeit(lambda : regSVDapprox(M,r), number=100))
    #print("Randomized SVD, rank {0}: {1} seconds".format(k, timeit(lambda : rSVDapprox(M,r, poweriterations, oversample), number=100))
    
def regcompressmatrix(M, k):
    M_type = M.dtype
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    b = regSVDapprox(M_stacked, k)
    #M_approx = imagerestack(M, b)
    M_approx = b.reshape(r, c, -1)
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        M_approx = np.clip(M_approx, 0, 255)
    else:
        M_approx = np.clip(img_approx, 0, 1)
    return M_approx.astype(M_type)

def regcompressvideo(M, k):
    M_shape = M.shape
    frames = M_shape[0]
    M_flat = M.reshape(-1, frames)
    M_flat_approx = regcompressmatrix(M,k)
    M_approx_video = M_flat_approx.reshape(M_shape)
    return M_approx_video

img2 = imagestack(img)
a = singularvalueplot(img2)



#a = regcompressmatrix(img, 2)
#plt.imshow(a)
#plt.show()