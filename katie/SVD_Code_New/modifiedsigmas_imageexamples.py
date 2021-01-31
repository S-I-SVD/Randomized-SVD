#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:18:14 2020

@author: katie
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from timeit import timeit
from mpl_toolkits import mplot3d
from svd import *

#images

img = imageio.imread('imageio:chelsea.png')

noodles = imageio.imread(r'/Users/katie/Downloads/noodles.png')

geese = imageio.imread(r'/Users/katie/Downloads/geese.png')

hawaii = imageio.imread(r'/Users/katie/Downloads/hawaii.png')

print(noodles.shape)
print(geese.shape)
print(hawaii.shape)

r1, c1 = img.shape[:2]
imgstacked = img.reshape(-1, c1)
imgrank = np.linalg.matrix_rank(imgstacked)

rnoodles, cnoodles = noodles.shape[:2]
noodlestacked = noodles.reshape(-1,cnoodles)
noodlerank = np.linalg.matrix_rank(noodlestacked)

rgeese, cgeese = geese.shape[:2]
geesestacked = geese.reshape(-1,cnoodles)
geeserank = np.linalg.matrix_rank(geesestacked)

rhawaii, chawaii = hawaii.shape[:2]
hawaiistacked = hawaii.reshape(-1,chawaii)
hawaiirank = np.linalg.matrix_rank(hawaiistacked)

fig, axs = plt.subplots(2,3)

axs[0,0].imshow(noodles)
axs[0,0].set_title("Original")


axs[0,1].imshow(geese)
axs[0,1].set_title("Original")


axs[0,2].imshow(hawaii)
axs[0,2].set_title("Original")

a3 = regcompressmatrix_modified(noodles,noodlerank,1.05,0)
axs[1,0].imshow(a3)
axs[1,0].set_title("n=1.05")

a4 = regcompressmatrix_modified(geese,geeserank,1.05,0)
axs[1,1].imshow(a4)
axs[1,1].set_title("n=1.05")

a5 = regcompressmatrix_modified(hawaii,hawaiirank,1.05,0)
axs[1,2].imshow(a5)
axs[1,2].set_title("n=1.05")

fig.tight_layout(pad=2.0)
plt.show()

