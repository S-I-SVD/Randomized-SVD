import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from timeit import timeit
from mpl_toolkits import mplot3d
from svd import *
from PIL import Image
import png


import sys

sys.path.append('/Users/katie/Documents/GitHub/Randomized-SVD/david')
import image_watermark_experiments.py

#import svd_tools as svdt
#import watermark as wm

#test_watermark_jain_mod_rotate()


#import ../../../david/watermark as watermarktools

sunset = imageio.imread('../res/sunset.png')
rainbow = imageio.imread('../res/rainbow.png')
view = imageio.imread('../res/view.png')
#
#def extraction_error(mat, watermark, scale):
#
#    img = np.asarray(Image.open('res/images/raccoon.jpg'))
#    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))
#    img_watermarked, watermarked_u, mat_s, watermarked_vh = embed_watermark(img, watermark, 
#            scale=scale)
#    watermark_extracted = extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh,
#            scale=scale, mode=mode, rank=rank)
#    
#      mat_watermarked, watermarked_u, mat_s_matrix, watermarked_vh watermarktool.embed_watermark(mat, watermark, scale)
def load_image(path, dtype=np.uint8):
    return np.asarray(Image.open(path)).astype(dtype)
