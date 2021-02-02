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

rainbow = imageio.imread('../res/rainbow.png')
view = imageio.imread('../res/view.png')

#example = regcompressmatrix_2(rainbow,rainbow.shape[1],2)
#example1 = regcompressmatrix_2(view,view.shape[1],2)

#Image.fromarray(example).convert('RGB').save('../out/example.png', 'PNG')
#Image.fromarray(example1).convert('RGB').save('../out/example1.png', 'PNG')

print(hawaii.shape)

def paper_sigmamod_multiply(M, a): #code producing figures in SIURO Paper
    modified = regcompressmatrix_2(M,M.shape[1],a)
    Image.fromarray(modified).convert('RGB').save('../out/sigmas_mod/sigmas_mod_multiply/sigmas_mod_multiply_{}.png'.format(a), 'PNG')
    
paper_sigmamod_multiply(view, 0.1)
paper_sigmamod_multiply(view, 0.5)
paper_sigmamod_multiply(view, 0.01)
paper_sigmamod_multiply(view, 0.75)
paper_sigmamod_multiply(view, 0.9)
paper_sigmamod_multiply(view, 1.1)
paper_sigmamod_multiply(view, 1.25)
paper_sigmamod_multiply(view, 1.5)
paper_sigmamod_multiply(view, 1.75)
paper_sigmamod_multiply(view, 2)
paper_sigmamod_multiply(view, 4)
paper_sigmamod_multiply(view, 3)