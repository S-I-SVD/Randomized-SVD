import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from timeit import timeit
from mpl_toolkits import mplot3d
from PIL import Image
import png
import svd_tools_copy as svdt
import image_tools_copy as it

#import ../../../david/watermark as watermarktools

#sunset = it.load_image('../res/sunset.png')
#rainbow = it.load_image('../res/rainbow.png')
#view = it.load_image('../res/view.png')

view = it.load_image('../res/view.jpg')
tree = it.load_image('../res/tree.jpg')



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

def extraction_error(scheme, scale=1):
    if scheme=='liutan':
        #img = np.asarray(Image.open('../res/view.jpg'))
        #watermark = np.asarray(Image.open('../res/tree.jpg'))
#        img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
#        plt.imshow(img_watermarked)
#        plt.show()
#        watermark_extracted = it.extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh,
#            scale=scale)
#        plt.imshow(watermark_extracted)
#        plt.show()
        print("what's up man")
    elif scheme =='jain':
        print("hi")
        #PUT JAIN CODE
    else: #JAINMOD
        print("hello")
    
#extraction_error('liutan')
#extraction_error('jain')
#extraction_error('jainmod')

def test_watermark(scale=1):
    img = np.asarray(Image.open('../res/view.jpg'))
    watermark = np.asarray(Image.open('../res/tree.jpg'))
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, 
            scale=scale)
    watermark_extracted = it.extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh,
            scale=scale)
    plt.imshow(watermark_extracted)
    plt.show()
    
test_watermark()

#def test_watermark(mode='randomized', rank=10, scale=1):
#    img = np.asarray(Image.open('res/images/raccoon.jpg'))
#    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))
#    img_watermarked, watermarked_u, mat_s, watermarked_vh = embed_watermark(img, watermark, 
#            scale=scale)
#    watermark_extracted = extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh,
#            scale=scale, mode=mode, rank=rank)
#    plt.imshow(watermark_extracted)
#    plt.show()