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
from PIL import Image
import png
from image_tools import *

#images

img = imageio.imread('imageio:chelsea.png')

noodles = imageio.imread(r'/Users/katie/Downloads/noodles.png')

geese = imageio.imread(r'/Users/katie/Downloads/geese.png')

hawaii = imageio.imread(r'/Users/katie/Downloads/hawaii.png')

example = regcompressmatrix_2(img, 50,4)

plt.imshow(example)

Image.fromarray(example).convert('RGB').save('../out/example.png', 'PNG')

print(hawaii.shape)

#def sigmamod_multiply(M, r, a):
    