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

def paper_sigmas_mod_multiply(M, a): #code producing figures in SIURO Paper
    modified = regcompressmatrix_2(M,M.shape[1],a)
    Image.fromarray(modified).convert('RGB').save('../out/sigmas_mod/sigmas_mod_multiply/sigmas_mod_multiply_{}.png'.format(a), 'PNG')
   
def paper_sigmas_mod_add(M,a):
    modified = regcompressmatrix_add(M,M.shape[1],a)
    Image.fromarray(modified).convert('RGB').save('../out/sigmas_mod/sigmas_mod_add/sigmas_mod_add_{}.png'.format(a), 'PNG')