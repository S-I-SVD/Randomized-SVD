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
import scipy

img = imageio.imread('imageio:chelsea.png')

def regularSVD(M):
    
    #stacking
    M_type = M.dtype
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    
    #create economy SVD for M
    U, S, VT = np.linalg.svd(M_stacked, full_matrices=False)
    S = np.diag(S)
    
    return U, S, VT

def regSVDapprox(M, r): 
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    S = np.diag(S)
    
    # Construct approximate image from U, S, VT with rank k
    M_approx = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    
    return M_approx

def rSVDapprox(M,r,poweriterations,oversample):
    U, S, VT = rSVD(M,poweriterations,oversample)
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
        approx[:,:,i] = regularSVD(colors[i], k)
    return approx  #returns the approximation of the original color image M

def colorstackingsvd(M,k):
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    

    M_rank = np.linalg.matrix_rank(M_stacked)
    M_approx_stacked = regularSVD(M_stacked, k)
    M_approx = M_approx_stacked.reshape(r, c, -1)
    return M_approx

def singularvalueplot(M): #plotting the singular values, can only be used on a stacked matrix
    U, S, VT = regularSVD(M)
    plt.semilogy(S)
    plt.title('Singular Values')
    plt.show()
    
def regcumsumplot(M): #how much of the original data is captured #"Singular Values: Cumulative Sum (Regular, Stacking)
    U, S, VT = regularSVD(M)
    plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))

def rcumsumplot(M,poweriterations,oversample): #how much of the original data is captured
    U, S, VT = rSVD(M,poweriterations,oversample)
    plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
    
def rSVD(M,poweriterations,oversample):
    #reshape M
    M_type = M.dtype
    r, c = M.shape[:2]
    M = M.reshape(-1, c)
    
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

def rcompressmatrix(M, k, poweriterations, oversample):
    #type
    M_type = M.dtype
    
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)

    #randomized
    U, S, VT = rSVD(M_stacked,poweriterations,oversample)
    
    #reconstructing matrix
    M_approx = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
    
    #reshaping back into original matrix
    M_approx = M_approx.reshape(r, c, -1)
    
    #dimension issues
    if M_approx.shape[2] == 1:
        M_approx.shape = M_approx.shape[:2] #should output a two dimensional matrix
    
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        M_approx = np.clip(M_approx, 0, 255)
    else:
        M_approx = np.clip(M_approx, 0, 1)
        
    return M_approx.astype(M_type)

def regcompressmatrix(M, k): 
    #type
    M_type = M.dtype
    
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    
    #rank
    M_rank = np.linalg.matrix_rank(M_stacked)
    
    #SVD
    U, S, VT = np.linalg.svd(M_stacked, full_matrices=False)
    S = np.diag(S)
    
    # Construct approximate image from U, S, VT with rank k 
    M_approx = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
    
    #reshaping back into original matrix
    M_approx = M_approx.reshape(r, c, -1)
    
    #dimension issues
    if M_approx.shape[2] == 1:
        M_approx.shape = M_approx.shape[:2] #should output a two dimensional matrix
        
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        M_approx = np.clip(M_approx, 0, 255)
    else:
        M_approx = np.clip(M_approx, 0, 1)
        
    return M_approx.astype(M_type)

def cSVD_II(M,rank,poweriterations,oversample):
    #oversampling
    l = rank + poweriterations
    
    #formatting the matrix
    M_type = M.dtype
    r, c = M.shape[:2]
    M = M.reshape(-1, c)
    m = M.shape[0]
    n = M.shape[1]
    
    
    #creating random l x m matrix
    psi = np.random.rand(l,m)
    
    #sketch input matrix
    Y = psi @ M
    
    #compute truncated SVD
    T, S_, V_ = np.linalg.svd(Y,rank)
    
    #update decomposition
    U, S, QT = np.linalg.svd(M @ V_)
    Q = QT.T
    
    #update right singular vectors
    V = V_ @ Q
    
    #since all the other SVD functions return VT, transpose V to get VT
    VT = V.T
    return U, S, VT

def cII_compressmatrix(M,rank,poweriterations,oversample):
    #type
    M_type = M.dtype
    
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    
    #rank
    M_rank = np.linalg.matrix_rank(M_stacked)
    
    #SVD
    U, S, VT = cSVD_II(M,rank,poweriterations,oversample)
    S = np.diag(S)
    M_approx = U[:,:rank] @ S[0:rank,:rank] @ VT[:rank,:]
    
    #reshaping back into original matrix
    M_approx = M_approx.reshape(r, c, -1)
    
    #dimension issues
    if M_approx.shape[2] == 1:
        M_approx.shape = M_approx.shape[:2] #should output a two dimensional matrix
        
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        M_approx = np.clip(M_approx, 0, 255)
    else:
        M_approx = np.clip(M_approx, 0, 1)
        
    return M_approx.astype(M_type)


def cSVD_I(M,rank,poweriterations,oversample):
    #oversampling
    l = rank + poweriterations
    
    #formatting the matrix
    M_type = M.dtype
    r, c = M.shape[:2]
    M = M.reshape(-1, c)
    m = M.shape[0]
    n = M.shape[1]
    
    #creating random l x m matrix
    psi = np.random.rand(l,m)
    
    #sketch input matrix
    Y = psi @ M
    
    #form smaller l x l matrix
    B = Y @ Y.T
    
    #ensure symmetry
    B = (1/2) * (B + B.T)
    
    #truncated eigendecomposition
    D,T = np.linalg.eig(B)
    T = T[0:rank]
    D = D[0:rank]
    D = np.diag(D)
    
    #rescale eigenvalues
    S_ = D ** 0.5
    
    #approximate right singular values
    V_ = Y.T @ T @ np.linalg.inv(S_)
    
    #approximate unscaled left singular values
    U_ = M @ V_
    
    #update left singular vectors and values
    U, S, QT = np.linalg.svd(U_)
    
    #transpose of Q
    Q = QT.T
    
    #update right singular values
    V = V_ @ Q
    
    #since all the other SVD functions return VT, transpose V to get VT
    VT = V.T
    return U, S, VT
    

def cI_compressmatrix(M,rank,poweriterations,oversample):
    #type
    M_type = M.dtype
    
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    
    #rank
    M_rank = np.linalg.matrix_rank(M_stacked)
    
    #SVD
    U, S, VT = cSVD_I(M,rank,poweriterations,oversample)
    S = np.diag(S)
    M_approx = U[:,:rank] @ S[0:rank,:rank] @ VT[:rank,:]
    
    #reshaping back into original matrix
    M_approx = M_approx.reshape(r, c, -1)
    
    #dimension issues
    if M_approx.shape[2] == 1:
        M_approx.shape = M_approx.shape[:2] #should output a two dimensional matrix
        
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        M_approx = np.clip(M_approx, 0, 255)
    else:
        M_approx = np.clip(M_approx, 0, 1)
        
    return M_approx.astype(M_type)  

def regcompressvideo(M, k):
    M_shape = M.shape
    frames = M_shape[0]
    M_flat = M.reshape(frames, -1)
    M_flat_approx = regcompressmatrix(M_flat,k)
    M_approx_video = M_flat_approx.reshape(M_shape) #takes two dimensional matrix and makes it in video format again
    return M_approx_video

def regcompressmatrix_2(M, k,j): #modified
    #type
    M_type = M.dtype
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    M_rank = np.linalg.matrix_rank(M_stacked)
    
    #SVD
    U, S, VT = np.linalg.svd(M_stacked, full_matrices=False)
    length_S = len(S)
    for i in range(length_S):
        S[i] = S[i]*j
    S = np.diag(S)
    
    # Construct approximate image from U, S, VT with rank k
    M_approx_stacked = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
    M_approx = M_approx_stacked.reshape(r, c, -1)
    
    #dimension issues
    if M_approx.shape[2] == 1:
        M_approx.shape = M_approx.shape[:2] #should output a two dimensional matrix
        
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        M_approx = np.clip(M_approx, 0, 255)
    else:
        M_approx = np.clip(M_approx, 0, 1)
        
    return M_approx.astype(M_type)

def regcompressmatrix_add(M, k,j): #modified, adding a scalar
    #type
    M_type = M.dtype
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    M_rank = np.linalg.matrix_rank(M_stacked)
    
    #SVD
    U, S, VT = np.linalg.svd(M_stacked, full_matrices=False)
    length_S = len(S)
    for i in range(length_S):
        S[i] = S[i] + j
    S = np.diag(S)
    
    # Construct approximate image from U, S, VT with rank k
    M_approx_stacked = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
    M_approx = M_approx_stacked.reshape(r, c, -1)
    
    #dimension issues
    if M_approx.shape[2] == 1:
        M_approx.shape = M_approx.shape[:2] #should output a two dimensional matrix
        
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        M_approx = np.clip(M_approx, 0, 255)
    else:
        M_approx = np.clip(M_approx, 0, 1)
        
    return M_approx.astype(M_type)

def regcompressmatrix_modified(M, k,a,b): 
    
    #type
    M_type = M.dtype
    
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    M_rank = np.linalg.matrix_rank(M_stacked)
    
    #SVD
    U, S, VT = np.linalg.svd(M_stacked, full_matrices=False)
    length_S = len(S)
    for i in range(length_S):
        S[i] = (S[i] ** a) + b
    S = np.diag(S)
    
    # Construct approximate image from U, S, VT with rank k 
    M_approx_stacked = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
    M_approx = M_approx_stacked.reshape(r, c, -1)
    
    #dimension issues
    if M_approx.shape[2] == 1:
        M_approx.shape = M_approx.shape[:2] #should output a two dimensional matrix
        
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        M_approx = np.clip(M_approx, 0, 255)
    else:
        M_approx = np.clip(M_approx, 0, 1)
        
    return M_approx.astype(M_type)

def regcompressmatrix_log(M, k,a): 
    M_type = M.dtype
    #stacking color channels
    r, c = M.shape[:2]
    M_stacked = M.reshape(-1, c)
    M_rank = np.linalg.matrix_rank(M_stacked)
    
    #SVD
    U, S, VT = np.linalg.svd(M_stacked, full_matrices=False)
    length_S = len(S)
    for i in range(length_S):
        S[i] = (np.log(S[i]+1)) ** a
    S = np.diag(S)
    
    # Construct approximate image from U, S, VT with rank k #color stacked
    M_approx_stacked = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
    M_approx = M_approx_stacked.reshape(r, c, -1)
    
    #dimension issues
    if M_approx.shape[2] == 1:
        M_approx.shape = M_approx.shape[:2] #should output a two dimensional matrix
        
    #overflow/underflow issues with color
    if np.issubdtype(M_type, np.integer):
        M_approx = np.clip(M_approx, 0, 255)
    else:
        M_approx = np.clip(M_approx, 0, 1)
        
    return M_approx.astype(M_type)

def modificationmatrix(img,k,a,scale):
    #"sharpening" images or altering them using the regcompressmatrix_log
    M_type = img.dtype
    r1, c1 = img.shape[:2]
    z = regcompressmatrix_log(img,k,a)
    rz, cz = z.shape[:2]
    zstring = z.reshape(-1)
    for i in range(0,len(zstring)): 
        if zstring[i] == 255:
            zstring[i] = 1
        else:
            zstring[i] = scale
    imgstring = img.reshape(-1)
    combo = np.empty_like(imgstring)
    for b in range(0, len(combo)):
        combo[b] = imgstring[b] * zstring[b]
    finalcombo = combo.reshape(rz, cz, -1)
    return(finalcombo)
    
def regcompressvideo_log(M, k,p):
    #more singular value modification
    M_shape = M.shape
    frames = M_shape[0]
    M_flat = M.reshape(frames, -1)
    M_flat_approx = regcompressmatrix_log(M_flat,k,p)
    M_approx_video = M_flat_approx.reshape(M_shape) #takes two dimensional matrix and makes it in video format again
    return M_approx_video

#plt.imshow(img)
#plt.show()
#a = regcompressmatrix(img,5)
#plt.imshow(a)
#plt.show()
#b = cI_compressmatrix(img,5,0,0)
#plt.imshow(b)
#plt.show()
#c = cII_compressmatrix(img,5,0,0)
#plt.imshow(c)
#plt.show()
