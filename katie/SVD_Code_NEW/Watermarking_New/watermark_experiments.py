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
        
def reversepad(watermark_extracted,original_watermark):
    sizes = original_watermark.shape
    watermark_extracted = watermark_extracted[:sizes[0],:sizes[1]]
    return watermark_extracted

def watermark_embed_liutan(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/liutan
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    if save=='no':
        return img_watermarked
    elif save=='yes':
        Image.fromarray(img_watermarked).convert('RGB').save('../out/watermarking/watermarked_image/liutan/watermarked_image_alpha_{}.png'.format(scale), 'PNG')
    
def watermark_extract_liutan(img, watermark, scale, save):
    #embeds watermark into image and then extracts the watermark. if save == 'yes', then it will save to out/res/watermark
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    watermark_extracted = it.extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh,
            scale=scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    if save=='no':
        return watermark_extracted_final
    elif save=='yes':
        Image.fromarray(watermark_extracted_final).convert('RGB').save('../out/watermarking/extracted_watermark/liutan/extracted_watermark_alpha_{}.png'.format(scale), 'PNG')
    
def watermark_embed_jain(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jain
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    if save=='no':
        return img_watermarked
    elif save=='yes':
        Image.fromarray(img_watermarked).convert('RGB').save('../out/watermarking/watermarked_image/jain/watermarked_image_alpha_{}.png'.format(scale), 'PNG')
    
        
def watermark_extract_jain(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jain
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    watermark_extracted = it.extract_watermark_jain(img_watermarked, img, watermark_vh, scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    if save=='no':
        return watermark_extracted_final
    elif save=='yes':
        Image.fromarray(watermark_extracted_final).convert('RGB').save('../out/watermarking/extracted_watermark/jain/extracted_watermark_alpha_{}.png'.format(scale), 'PNG')
    
def watermark_embed_jain_mod(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jainmod
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    if save=='no':
        return img_watermarked
    elif save=='yes':
        Image.fromarray(img_watermarked).convert('RGB').save('../out/watermarking/watermarked_image/jainmod/watermarked_image_alpha_{}.png'.format(scale), 'PNG')
    
def watermark_extract_jain_mod(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jainmod
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    watermark_extracted = it.extract_watermark_jain_mod(img_watermarked, img, watermark_vh, scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    if save=='no':
        return watermark_extracted_final
    elif save=='yes':
        Image.fromarray(watermark_extracted_final).convert('RGB').save('../out/watermarking/extracted_watermark/jainmod/extracted_watermark_alpha_{}.png'.format(scale), 'PNG')
        


watermark_extract_liutan(view, tree, 0.05, 'yes')
watermark_extract_liutan(view, tree, 0.25, 'yes')
watermark_extract_liutan(view, tree, 0.5, 'yes')
watermark_extract_liutan(view, tree, 0.75, 'yes')
watermark_extract_liutan(view, tree, 1, 'yes')


watermark_embed_liutan(view, tree, 0.05, 'yes')
watermark_embed_liutan(view, tree, 0.25, 'yes')
watermark_embed_liutan(view, tree, 0.5, 'yes')
watermark_embed_liutan(view, tree, 0.75, 'yes')
watermark_embed_liutan(view, tree, 1, 'yes')

watermark_embed_jain(view, tree, 0.05, 'yes')
watermark_embed_jain(view, tree, 0.25, 'yes')
watermark_embed_jain(view, tree, 0.5, 'yes')
watermark_embed_jain(view, tree, 0.75, 'yes')
watermark_embed_jain(view, tree, 1, 'yes')

watermark_extract_jain(view, tree, 0.05, 'yes')
watermark_extract_jain(view, tree, 0.25, 'yes')
watermark_extract_jain(view, tree, 0.5, 'yes')
watermark_extract_jain(view, tree, 0.75, 'yes')
watermark_extract_jain(view, tree, 1, 'yes')

watermark_embed_jain_mod(view, tree, 0.05, 'yes')
watermark_embed_jain_mod(view, tree, 0.25, 'yes')
watermark_embed_jain_mod(view, tree, 0.5, 'yes')
watermark_embed_jain_mod(view, tree, 0.75, 'yes')
watermark_embed_jain_mod(view, tree, 1, 'yes')

watermark_extract_jain_mod(view, tree, 0.05, 'yes')
watermark_extract_jain_mod(view, tree, 0.25, 'yes')
watermark_extract_jain_mod(view, tree, 0.5, 'yes')
watermark_extract_jain_mod(view, tree, 0.75, 'yes')
watermark_extract_jain_mod(view, tree, 1, 'yes')