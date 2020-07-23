#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:52:21 2020

@author: jenzyy
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pydub 
import scipy.signal as sig


'''
# set wd
import os
os. chdir('./Documents/GitHub/Randomized-SVD/jennifer')
'''

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")
    
def watermark_image(im, W, a):
    rows,cols = im.shape[:2]
    U,S,V = np.linalg.svd(im,full_matrices = False)
    Wp = np.pad(W,[(0, rows - W.shape[0]), (0, rows - W.shape[1])])
    Aw = np.diag(S)+a*Wp
    Uw,Sw,Vw = np.linalg.svd(Aw,full_matrices = True)
    marked = U @ np.diag(Sw) @ V
    return marked, Uw, S, Vw

def watermark_extract(marked, Uw, S,Vw, a):
    Um, Sm, Vm = np.linalg.svd(marked)
    M = (Uw @ np.diag(Sm) @ Vw - np.diag(S))/a
    #rows = len(S)
    #Mp = np.pad(M,[(0, M.shape[0]- rows), (0, M.shape[1] - rows)])
    return M

def watermark(im, W, a):
    rows,cols = im.shape[:2]
    U,S,V = np.linalg.svd(im,full_matrices = False)
    Wp = np.pad(W,[(0, rows - W.shape[0]), (0, rows - W.shape[1])])
    Aw = np.diag(S)+a*Wp
    Uw,Sw,Vw = np.linalg.svd(Aw,full_matrices = True)
    marked = U @ np.diag(Sw) @ V
    # extract watermark
    Um, Sm, Vm = np.linalg.svd(marked)
    M = (Uw @ np.diag(Sm) @ Vw - np.diag(S))/a
    Mrow, Mcol = W.shape
    M = M[:Mrow, :Mcol]
    return marked, M

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# load mp3
sr, x = read('bach.mp3')
W_sr, W_x = read('news3.mp3')

f,t,mat = sig.stft(x[:,0])
f,t,W_mat = sig.stft(W_x[:,0])

# Watermark
marked, Uw, S, Vw = watermark_image(mat, W_mat,0.1)

# reformat
ts,new = sig.istft(marked)
write("bach_w.mp3",sr,new)

# extract watermark
M = watermark_extract(marked, Uw, S, Vw, 0.1)
ts,new_marked = sig.istft(M)
write("news3_e.mp3",W_sr, new_marked)

# image as watermark
Wg = rgb2gray(imageio.imread("dog.jpg"))
marked, M = watermark(mat, Wg, 0.1)

# scalar
for a in np.arange(0.1,2,0.1):
    marked, M = watermark(mat, Wg, a)
    diff = np.linalg.norm(M-Wg)
    plt.scatter(a,diff)
    plt.xlabel('Scalar')
    plt.ylabel('Norm Difference')
    plt.show()