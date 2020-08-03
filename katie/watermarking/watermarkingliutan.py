#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:52:05 2020

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

coffee = imageio.imread(r'/Users/katie/Downloads/coffee.png')
cat = imageio.imread(r'/Users/katie/opt/anaconda3/pkgs/scikit-image-0.15.0-py37h0a44026_0/lib/python3.7/site-packages/skimage/data/eye.png')
cat=cat[:,:,:-1]
randommatrix = np.random.rand(600,600)

def watermarkimage_liutan(M, a, W): #M is matrix, a is scaling factor, W is watermark
    #SVD
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    S = np.diag(S) 
    W_padded = padimage(W,S)
    newmatrix = S + (a * W_padded)
    U_W, S_W, VT_W = np.linalg.svd(newmatrix, full_matrices=False)
    S_W = np.diag(S_W)
    A_W = U @ S_W @ VT
    return A_W

def extractwatermark_liutan(A_W,U_W,S,VT_W,a):
    U, S_W, VT = np.linalg.svd(A_W, full_matrices=False)
    S_W = np.diag(S_W)
    D = U_W @ S_W @ VT_W
    a_ = 1/a
    new = D - S
    W = a_ * new
    return W
    
def importantwatermarkelements_liutan(M,a,W):
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    S = np.diag(S) 
    W_padded = padimage(W,S)
    newmatrix = S + (a * W_padded)
    U_W, S_W, VT_W = np.linalg.svd(newmatrix, full_matrices=False)
    return U_W, S, VT_W

def watermark_liutan(M, a, W,random):
    #enter random=1 if using a random matrix as the watermark, else random=0 if it's an image watermark
    #formatting
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    
    #embedding
    watermarkedimagestacked = watermarkimage_liutan(M_stacked, a, W_stacked)
    watermarkedimage = watermarkedimagestacked.reshape(r_M, c_M, -1)
    if np.issubdtype(M_type, np.integer):
        watermarkedimage = np.clip(watermarkedimage, 0, 255)
    else:
        watermarkedimage = np.clip(watermarkedimage, 0, 1)
    watermarkedimage = watermarkedimage.astype(M_type)
    plt.title("Image with watermark: a = {0}".format(a))
    plt.imshow(watermarkedimage)
    plt.show()
    
    #extraction
    U_W, S, VT_W = importantwatermarkelements(M_stacked, a, W_stacked)
    extractedwatermark = extractwatermark(watermarkedimagestacked,U_W,S,VT_W,a)
    extractedwatermark = reversepad(extractedwatermark,W_stacked)
    if random == 0:
        extractedwatermark = extractedwatermark.reshape(r_W,c_W,-1)
        if np.issubdtype(W_type, np.integer):
            extractedwatermark = np.clip(extractedwatermark, 0, 255)
        else:
            extractedwatermark = np.clip(extractedwatermark, 0, 1)
        extractedwatermark = extractedwatermark.astype(W_type)
    else:
        extractedwatermark = extractedwatermark.reshape(-1,c_W)
    plt.title("Extracted watermark (Liu & Tan)")
    plt.imshow(extractedwatermark)
    plt.show()
    
def watermarkonly_liutan(M, a, W): 
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    watermarkedimagestacked = watermarkimage_liutan(M_stacked, a, W_stacked)
    watermarkedimage = watermarkedimagestacked.reshape(r_M, c_M, -1)
    if np.issubdtype(M_type, np.integer):
        watermarkedimage = np.clip(watermarkedimage, 0, 255)
    else:
        watermarkedimage = np.clip(watermarkedimage, 0, 1)
    watermarkedimage = watermarkedimage.astype(M_type)
    return(watermarkedimage)

def perceptibility_liutan(M,a,W): #returns difference between watermarked image and original
    #formatting
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    #watermark embedding
    watermarkedimagestacked = watermarkimage(M_stacked, a, W_stacked)
    watermarkedimage = watermarkedimagestacked.reshape(r_M, c_M, -1)
    watermarkedimagestacked = watermarkedimage.reshape(-1,c_M)
    #error
    error = (np.linalg.norm((M_stacked-watermarkedimagestacked),ord='fro'))/(np.linalg.norm(M_stacked,ord='fro'))
    return error
    
def lowrankwatermarknormdifference_liutan(M, a, W,k,showimage,random): 
    #if showimage=1, it will return the watermark extracted from the low rank image. otherwise, it will return the error.
    #if random=1, you're using a random matrix. If using an image matrix, enter random=0.
    #formatting
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    #watermarking
    watermarkedimagestacked = watermarkimage_liutan(M_stacked, a, W_stacked)
    U_W, S, VT_W = importantwatermarkelements_liutan(M_stacked, a, W_stacked)
    #low rank distortion
    lowrankwatermarkedimage = lowrankapprox(watermarkedimagestacked,k)
    lowrankwatermarkedimage = lowrankwatermarkedimage.reshape(-1,c_M)
    extractedwatermark = extractwatermark_liutan(lowrankwatermarkedimage,U_W,S,VT_W,a)
    extractedwatermark = reversepad(extractedwatermark,W_stacked)
    if showimage == 1:
        if random==0:
            extractedwatermark = extractedwatermark.reshape(r_W,c_W,-1)
            if np.issubdtype(W_type, np.integer):
                extractedwatermark = np.clip(extractedwatermark, 0, 255)
            else:
                extractedwatermark = np.clip(extractedwatermark, 0, 1)
            extractedwatermark = extractedwatermark.astype(W_type)
        else:
            extractedwatermark = extractedwatermark.reshape(-1,c_W)
        return extractedwatermark
    else:
        extractedwatermark = extractedwatermark.reshape(-1,c_W)
        error = (np.linalg.norm((extractedwatermark-W_stacked),ord='fro'))/(np.linalg.norm(W_stacked,ord='fro'))
        return error
    
def extractedwatermarkdifference_liutan(M, a, W):
    #formatting
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    #embedding
    watermarkedimagestacked = watermarkimage_liutan(M_stacked, a, W_stacked)
    U_W, S, VT_W = importantwatermarkelements_liutan(M_stacked, a, W_stacked)
    extractedwatermark = extractwatermark_liutan(watermarkedimagestacked,U_W,S,VT_W,a)
    extractedwatermark = reversepad(extractedwatermark,W_stacked)
    #reshaping 
    extractedwatermark = extractedwatermark.reshape(-1,c_W)
    error = (np.linalg.norm(extractedwatermark - W_stacked,ord='fro'))/(np.linalg.norm(W_stacked,ord='fro'))
    return error

def extractedwatermarkdifferencecropped_liutan(M, a, W,cropped,showimage,random):
    #if showimage=1, it will return the watermark extracted from the low rank image. otherwise, it will return the error.
    #if random=1, you're using a random matrix. If using an image matrix, enter random=0.
    #formatting
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    r_cropped, c_cropped = M.shape[:2]
    cropped_stacked = cropped.reshape(-1,c_cropped)
    #watermarking the image
    watermarkedimagestacked = watermarkimage_liutan(M_stacked, a, W_stacked)
    U_W, S, VT_W = importantwatermarkelements_liutan(M_stacked, a, W_stacked)
    extractedwatermark = extractwatermark_liutan(watermarkedimagestacked,U_W,S,VT_W,a)
    extractedwatermark = reversepad(extractedwatermark,W_stacked)
    extractedwatermark = extractedwatermark.reshape(-1,c_W)
    #padding the cropped image to give it the same dimensions as M_stacked
    cropped_padded = padimage(cropped_stacked,M_stacked)
    #extracting the watermark
    extractedwatermarkcropped = extractwatermark(cropped_padded,U_W,S,VT_W,a)
    extractedwatermarkcropped = reversepad(extractedwatermarkcropped,W_stacked)
    #what to return
     if showimage==1:
        if random == 0:
            extractedwatermarkcroppedimage = extractedwatermarkcropped.reshape(r_W,c_W,-1)
            if np.issubdtype(W_type, np.integer):
                extractedwatermarkcroppedimage = np.clip(extractedwatermarkcroppedimage, 0, 255)
            else:
                extractedwatermarkcroppedimage = np.clip(extractedwatermarkcroppedimage, 0, 1)
            extractedwatermarkcroppedimage = extractedwatermarkimage.astype(W_type)
        else:
            extractedwatermarkcroppedimage = extractedwatermarkcropped.reshape(-1,c_W)
        return extractedwatermarkcroppedimage
    else:
        extractedwatermarkcropped = extractedwatermarkcropped.reshape(-1,c_W)
        error = (np.linalg.norm((extractedwatermark - extractedwatermarkcropped),ord='fro'))/(np.linalg.norm(extractedwatermark,ord='fro'))
        return error
    
def differentalphaslowrank_liutan(M,W,random):
    #random=1 if it's a random matrix, random=0 otherwise
    differentalpha = (0.05,0.1,0.15,0.2,0.25,0.5,0.75)
    ranks = np.arange(1,118,1)
    differences0 = []
    for rank in ranks:
        difference0 = lowrankwatermarknormdifference(M, differentalpha[0], W,rank,0,random)
        differences0.append(difference0)
    differences1 = []
    for rank in ranks:
        difference1 = lowrankwatermarknormdifference(M, differentalpha[1], W,rank,0,random)
        differences1.append(difference1)
    differences2 = []
    for rank in ranks:
        difference2 = lowrankwatermarknormdifference(M, differentalpha[2], W,rank,0,random)
        differences2.append(difference2)
    differences3 = []
    for rank in ranks:
        difference3 = lowrankwatermarknormdifference(M, differentalpha[3], W,rank,0,random)
        differences3.append(difference3)
    differences4 = []
    for rank in ranks:
        difference4 = lowrankwatermarknormdifference(M, differentalpha[4], W,rank,0,random)
        differences4.append(difference4)
    differences5 = []
    for rank in ranks:
        difference5 = lowrankwatermarknormdifference(M, differentalpha[5], W,rank,0,random)
        differences5.append(difference5)
    differences6 = []
    for rank in ranks:
        difference6 = lowrankwatermarknormdifference(M, differentalpha[6], W,rank,0,random)
        differences6.append(difference6)
    plt.plot(differences0,label="a = {0}".format(differentalpha[0]))
    plt.plot(differences1,label="a = {0}".format(differentalpha[1]))
    plt.plot(differences2,label="a = {0}".format(differentalpha[2]))
    plt.plot(differences3,label="a = {0}".format(differentalpha[3]))
    plt.plot(differences4,label="a = {0}".format(differentalpha[4]))
    plt.plot(differences5,label="a = {0}".format(differentalpha[5]))
    plt.plot(differences6,label="a = {0}".format(differentalpha[6]))
    plt.xlabel('Rank')
    plt.ylabel('Error (Frobenius Norm)')
    if random == 1:
        plt.title('Watermark Extraction Error for Low Rank Approximations (Liu & Tan) \n Random Matrix')
    else:
        plt.title('Watermark Extraction Error for Low Rank Approximations (Liu & Tan)')
    plt.legend()
    plt.show()

def extractedwatermarkonly_cropped_liutan(M, a, W,cropped,random):
    #random=1 if it's a random matrix watermark, random=0 if image watermark
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    r_cropped, c_cropped = M.shape[:2]
    cropped_stacked = cropped.reshape(-1,c_cropped)
    #watermarking the image
    watermarkedimagestacked = watermarkimage_liutan(M_stacked, a, W_stacked)
    U_W, S, VT_W = importantwatermarkelements_liutan(M_stacked, a, W_stacked)
    extractedwatermark = extractwatermark_liutan(watermarkedimagestacked,U_W,S,VT_W,a)
    #extracted watermark - the way it's "supposed" to be
    extractedwatermark = reversepad(extractedwatermark,W_stacked)
    extractedwatermark = extractedwatermark.reshape(-1,c_W)
    #padding the cropped image to give it the same dimensions as M_stacked
    cropped_padded = padimage(cropped_stacked,M_stacked)
    #extracting the watermark
    extractedwatermarkcropped = extractwatermark_liutan(cropped_padded,U_W,S,VT_W,a)
    extractedwatermarkcropped = reversepad(extractedwatermarkcropped,W_stacked)
    if random==0:
        extractedwatermarkimage = extractedwatermarkcropped.reshape(r_W,c_W,-1)
        if np.issubdtype(W_type, np.integer):
            extractedwatermarkimage = np.clip(extractedwatermarkimage, 0, 255)
        else:
            extractedwatermarkimage = np.clip(extractedwatermarkimage, 0, 1)
        extractedwatermarkimage = extractedwatermarkimage.astype(W_type)
    else:
        extractedwatermarkimage = extractedwatermarkcropped.reshape(-1,c_W)
    return extractedwatermarkimage

def cropplots(M,a,W,random):
    #random=1 if it's a random matrix watermark, random=0 if image watermark
    M_W = watermarkonly_liutan(M,a,W)
    fig, axs = plt.subplots(4,2,figsize = (7,10))
    #50 columns removed from left
    M_left = M_W[:,100:,:]
    axs[0,0].imshow(coffeeleft)
    axs[0,0].set_title("100 Columns Removed from Left")
    M_left_extracted = extractedwatermarkdifferencecropped_liutan(M, a, W,M_left,showimage=1,random)
    axs[0,1].imshow(M_left_extracted)
    axs[0,1].set_title("Extracted Watermark")
    #50 columns removed from right
    M_right = M_W[:,:-100,:]
    axs[1,0].imshow(coffeeright)
    axs[1,0].set_title("100 Columns Removed from Right")
    M_right_extracted = extractedwatermarkdifferencecropped_liutan(M, a, W,M_right,showimage=1,random)
    axs[1,1].imshow(M_right_extracted)
    axs[1,1].set_title("Extracted Watermark")
    #50 rows removed from bottom
    M_bottom = M_W[:-100,:,:]
    axs[2,0].imshow(coffeebottom)
    axs[2,0].set_title("100 Rows Removed from Bottom")
    M_bottom_extracted = extractedwatermarkdifferencecropped_liutan(M, a, W,M_bottom,showimage=1,random)
    axs[2,1].imshow(M_bottom_extracted)
    axs[2,1].set_title("Extracted Watermark")
    #50 rows removed from top
    M_top = M_W[100:,:,:]
    axs[3,0].imshow(M_top)
    axs[3,0].set_title("100 Rows Removed from Top")
    M_top_extracted = extractedwatermarkdifferencecropped_liutan(M, a, W,M_top,showimage=1,random)
    axs[3,1].imshow(M_top_extracted)
    axs[3,1].set_title("Extracted Watermark")
    fig.tight_layout(pad=1.0)
    plt.show()
    
def lowrankwatermarkedimage_liutan(M, a, W,k):
    #formatting
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    watermarkedimagestacked = watermarkimage_liutan(M_stacked, a, W_stacked)
    U_W, S, VT_W = importantwatermarkelements_liutan(M_stacked, a, W_stacked)
    lowrankwatermarkedimage = lowrankapprox(watermarkedimagestacked,k)
    watermarkedimage = lowrankwatermarkedimage.reshape(r_M, c_M, -1)
    if np.issubdtype(M_type, np.integer):
        watermarkedimage = np.clip(watermarkedimage, 0, 255)
    else:
        watermarkedimage = np.clip(watermarkedimage, 0, 1)
    watermarkedimage = watermarkedimage.astype(M_type)
    return watermarkedimage

def extractedwatermark_liutan(M, a, W,random):
    #formatting
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    #watermarking
    watermarkedimagestacked = watermarkimage_liutan(M_stacked, a, W_stacked)
    U_W, S, VT_W = importantwatermarkelements_liutan(M_stacked, a, W_stacked)
    extractedwatermark = extractwatermark_liutan(watermarkedimagestacked,U_W,S,VT_W,a)
    extractedwatermark = reversepad(extractedwatermark,W_stacked)
    if random == 0
        extractedwatermark = extractedwatermark.reshape(r_W,c_W,-1)
        if np.issubdtype(W_type, np.integer):
            extractedwatermark = np.clip(extractedwatermark, 0, 255)
        else:
            extractedwatermark = np.clip(extractedwatermark, 0, 1)
        extractedwatermark = extractedwatermark.astype(W_type)
    else:
        extractedwatermark = extractedwatermark.reshape(-1,c_W)
    return extractedwatermark

def differentscalingfactorplots_liutan(M,W,random):
    fig, axs = plt.subplots(4,2,figsize = (7,10))
    #scaling factor 0.05
    M005= watermarkonly_liutan(M, 0.05,W)
    axs[0,0].imshow(M005)
    axs[0,0].set_title("Watermarked Image: a = 0.05")
    W005 = extractedwatermark_liutan(M, 0.05,W,random)
    if random==1:
        axs[0,1].imshow(W005,cmap='gray')
    else:
        axs[0,1].imshow(W005)
    axs[0,1].set_title("Extracted Watermark")
    #scaling factor 0.5
    M050 = watermarkonly_liutan(M, 0.5,W)
    axs[1,0].imshow(M050)
    axs[1,0].set_title("Watermarked Image: a = 0.5")
    W050 = extractedwatermark_liutan(M, 0.5,W,random)
    if random==1:
        axs[1,1].imshow(W050,cmap='gray')
    else:
        axs[1,1].imshow(W050)
    axs[1,1].set_title("Extracted Watermark")
    #scaling factor 0.75
    M150 = watermarkonly_liutan(M, 1.5,W)
    axs[2,0].imshow(M150)
    axs[2,0].set_title("Watermarked Image: a = 1.5")
    W150 = extractedwatermark_liutan(M, 1.5,W,random)
    if random==1:
        axs[2,1].imshow(W150,cmap='gray')
    else:
        axs[2,1].imshow(W150)
    axs[2,1].set_title("Extracted Watermark")
    #scaling factor 2
    M200 = watermarkonly_jainmod(M, 2,W)
    axs[3,0].imshow(M200)
    axs[3,0].set_title("Watermarked Image: a = 2")
    W200 = extractedwatermark_liutan(M, 2,W,random)
    if random==1:
        axs[3,1].imshow(W200,cmap='gray')
    else:
        axs[3,1].imshow(W200)
    axs[3,1].set_title("Extracted Watermark")
    fig.tight_layout(pad=1.0)
    plt.show()  
    
def lowranknormdifference_liutan(M, a, W,k,showimage,random):
    #if showimage=1, it will return the watermark extracted from the low rank image. otherwise, it will return the error.
    #if random=1, you're using a random matrix. If using an image matrix, enter random=0.
    #formatting
    r_M, c_M = M.shape[:2]
    M_stacked = M.reshape(-1,c_M)
    M_type = M.dtype
    r_W, c_W = W.shape[:2]
    W_stacked = W.reshape(-1,c_W)
    W_type = W.dtype
    #watermarking
    watermarkedimagestacked = watermarkimage_jainmod(M_stacked, a, W_stacked)
    U_W, S, VT_W = importantwatermarkelements_liutan(M_stacked, a, W_stacked)
    lowrankwatermarkedimage = lowrankapprox(watermarkedimagestacked,k)
    watermarkedimage = lowrankwatermarkedimage.reshape(r_M, c_M, -1)
    lowrankwatermarkedimage = watermarkedimage.reshape(-1,c_M)
    extractedwatermark = extractwatermark_liutan(lowrankwatermarkedimage,U_W,S,VT_W,a)
    extractedwatermark = reversepad(extractedwatermark,W_stacked)
    if showimage == 1:
        if random==0:
            extractedwatermark = extractedwatermark.reshape(r_W,c_W,-1)
            if np.issubdtype(W_type, np.integer):
                extractedwatermark = np.clip(extractedwatermark, 0, 255)
            else:
                extractedwatermark = np.clip(extractedwatermark, 0, 1)
            extractedwatermark = extractedwatermark.astype(W_type)
        else:
            extractedwatermark = extractedwatermark.reshape(-1,c_W)
        return extractedwatermark
    else:
        extractedwatermark = extractedwatermark.reshape(-1,c_W)
        error = (np.linalg.norm((extractedwatermark-W_stacked),ord='fro'))/(np.linalg.norm(W_stacked,ord='fro'))
        return error
    
def lowrankplots_liutan(M,a,W,random):
    #if random=1, you're using a random matrix. If using an image matrix, enter random=0.
    fig, axs = plt.subplots(4,2,figsize = (7,10))
    #rank 1
    M_rank1 = lowrankwatermarkedimage_liutan(M, a, W,1)
    axs[0,0].imshow(M_rank1)
    axs[0,0].set_title("Watermarked Image: Rank 1")
    W_rank1 = lowranknormdifference_liutan(M, a, W,1,1,random)
    if random==1:
        axs[0,1].imshow(W_rank1,cmap='gray')
    else:
        axs[0,1].imshow(W_rank1)
    axs[0,1].set_title("Extracted Watermark")
    #rank 10
    M_rank10 = lowrankwatermarkedimage_liutan(M, a, W,10)
    axs[1,0].imshow(M_rank10)
    axs[1,0].set_title("Watermarked Image: Rank 10")
    W_rank10 = lowranknormdifference_liutan(M, a, W,10,1,random)
    if random==1:
        axs[1,1].imshow(W_rank10,cmap='gray')
    else:
        axs[1,1].imshow(W_rank10)
    axs[1,1].set_title("Extracted Watermark")
    #rank 100
    M_rank100 = lowrankwatermarkedimage_liutan(M, a, W,100)
    axs[2,0].imshow(M_rank100)
    axs[2,0].set_title("Watermarked Image: Rank 100")
    W_rank100 = lowranknormdifference_liutan(M, a, W,100,1,random)
    if random==1:
        axs[2,1].imshow(W_rank100,cmap='gray')
    else:
        axs[2,1].imshow(W_rank100)
    axs[2,1].set_title("Extracted Watermark")
    #rank 400
    M_rank400 = lowrankwatermarkedimage_liutan(M, a, W,400)
    axs[3,0].imshow(M_rank400)
    axs[3,0].set_title("Watermarked Image: Rank 400")
    W_rank400 = lowranknormdifference_liutan(M, a, W,400,1,random)
    if random==1:
        axs[3,1].imshow(W_rank400,cmap='gray')
    else:
        axs[3,1].imshow(W_rank400)
    axs[3,1].set_title("Extracted Watermark")
    fig.tight_layout(pad=1.0)
    plt.show()
