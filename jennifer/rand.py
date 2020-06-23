# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:00:16 2020

@author: jenni
"""
from matplotlib.pyplot import imread
import imageio 
import matplotlib.pyplot as plt
import numpy as np

# set wd
#os. chdir('.\Desktop\Summer@ICERM\Week 2')

# import image
im = imageio.imread("sky.jpg")

# rgb
red = im[:,:,0]
green = im[:,:,1]
blue = im[:,:,2]


'''
X is input matrix
r is rank
q is power iteration
p is oversampling
'''
    
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

approx = np.empty_like(im)
q = 1
p = 5
for r in[1,10,100,849]:
    approx[:,:,0]=rand_svd(red, r, q, p)
    approx[:,:,1]=rand_svd(green, r, q, p)
    approx[:,:,2]=rand_svd(blue, r, q, p)
    plt.imshow(approx)
    plt.show()
    
    
    