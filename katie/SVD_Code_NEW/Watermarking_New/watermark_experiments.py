import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from timeit import timeit
from mpl_toolkits import mplot3d
from svd import *
from PIL import Image
import png
import svd_tools_copy as svdt
import image_tools_copy as it

#import ../../../david/watermark as watermarktools

#sunset = it.load_image('../res/sunset.png')
#rainbow = it.load_image('../res/rainbow.png')
#view = it.load_image('../res/view.png')

view = it.load_image('../res/view.png')
tree = it.load_image('../res/tree.png')

#sunset_f = sunset.astype(np.float64)
#rainbow_f = rainbow.astype(np.float64)
#
#raccoon = load_image('res/public/raccoon.jpg')
#fox = load_image('res/public/fox.jpg')
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


#EXTRACTION ERROR = NORM(ORIGINAL WATERMARK - EXTRACTED WATERMARK)
#1. COMPUTE EMBEDDING AND EXTRACTION
#2. COMPUTE NORM(ORIGINAL WATERMARK - EXTRACTED WATERMARK)/NORM(ORIGINAL WATERMARK)

def extraction_error(scheme):
    if scheme=='liutan':
        img_watermarked, u, s, vh = it.embed_watermark(view, tree)
    elif scheme =='jain':
        print("hi")
        #PUT JAIN CODE
    else: #JAINMOD
        print("hello")
    
extraction_error('liutan')
extraction_error('jain')
extraction_error('jainmod')