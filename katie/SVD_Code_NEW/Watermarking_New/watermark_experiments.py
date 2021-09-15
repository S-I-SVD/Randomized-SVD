import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import imageio
from timeit import timeit
from mpl_toolkits import mplot3d
from PIL import Image
#import png
import svd_tools_copy as svdt
import image_tools_copy as it

#import ../../../david/watermark as watermarktools

#sunset = it.load_image('../res/sunset.png')
#rainbow = it.load_image('../res/rainbow.png')
#view = it.load_image('../res/view.png')

view = it.load_image('../res/view.jpg')
tree = it.load_image('../res/tree.jpg')

plt.rcParams['font.size'] = '18'


def sv_plot_save(img, fname): #plotting the singular values, can only be used on a stacked matrix
    #formatting
    img = img.astype(np.float64)
    
    #stacking color channels
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    
    u, s, v = np.linalg.svd(img_stacked, full_matrices=False)
    
    plt.plot(s)
    plt.savefig(fname)

#EXTRACTION ERROR = NORM(ORIGINAL WATERMARK - EXTRACTED WATERMARK)
#1. COMPUTE EMBEDDING AND EXTRACTION
#2. COMPUTE NORM(ORIGINAL WATERMARK - EXTRACTED WATERMARK)/NORM(ORIGINAL WATERMARK)

def reversepad(watermark_extracted,original_watermark):
    sizes = original_watermark.shape
    watermark_extracted = watermark_extracted[:sizes[0],:sizes[1]]
    return watermark_extracted

def reversepad3d(watermark_extracted,original_watermark):
    sizes = original_watermark.shape
    watermark_extracted = watermark_extracted[:sizes[0],:sizes[1],:sizes[2]]
    return watermark_extracted

def watermark_embed_liutan(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/liutan
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    if save=='no':
        return img_watermarked
    elif save=='yes':
        it.save_image(img_watermarked,'../out/watermarking/watermarked_image/liutan/watermarked_image_alpha_{}.png'.format(scale))
        #Image.fromarray(img_watermarked,'RGB').save('../out/watermarking/watermarked_image/liutan/watermarked_image_alpha_{}.png'.format(scale), 'PNG')
    
def watermark_extract_liutan(img, watermark, scale, save):
    #embeds watermark into image and then extracts the watermark. if save == 'yes', then it will save to out/res/watermark
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    watermark_extracted = it.extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh,
            scale=scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    watermark_extracted_final = watermark_extracted_final.astype(np.int32)
    if save=='no':
        return watermark_extracted_final
    elif save=='yes':
        it.save_image(watermark_extracted_final,'../out/watermarking/extracted_watermark/liutan/extracted_watermark_alpha_{}.png'.format(scale))
    
def watermark_embed_jain(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jain
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    if save=='no':
        return img_watermarked
    elif save=='yes':
        it.save_image(img_watermarked,'../out/watermarking/watermarked_image/jain/watermarked_image_alpha_{}.png'.format(scale))
    
def watermark_extract_jain(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jain
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    watermark_extracted = it.extract_watermark_jain(img_watermarked, img, watermark_vh, scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    watermark_extracted_final = watermark_extracted_final.astype(np.int32)
    if save=='no':
        return watermark_extracted_final
    elif save=='yes':
        it.save_image(watermark_extracted_final,'../out/watermarking/extracted_watermark/jain/extracted_watermark_alpha_{}.png'.format(scale))
    
def watermark_embed_jain_mod(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jainmod
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    if save=='no':
        return img_watermarked
    elif save=='yes':
        it.save_image(img_watermarked,'../out/watermarking/watermarked_image/jainmod/watermarked_image_alpha_{}.png'.format(scale))
    
def watermark_extract_jain_mod(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jainmod
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    watermark_extracted = it.extract_watermark_jain_mod(img_watermarked, img, watermark_vh, scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    watermark_extracted_final = watermark_extracted_final.astype(np.int32)
    if save=='no':
        return watermark_extracted_final
    elif save=='yes':
        it.save_image(watermark_extracted_final,'../out/watermarking/extracted_watermark/jainmod/extracted_watermark_alpha_{}.png'.format(scale))
        
def perceptibility_liutan(img, watermark, scale):
    #watermarked image
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    #stacking watermarked image
    img_watermarked = img_watermarked.astype(np.int32)

    img_watermarked_rows, img_watermarked_columns = img_watermarked.shape[:2] 
    img_watermarked_stacked = img_watermarked.reshape(img_watermarked_rows, -1)
    #stacking image
    img = img.astype(np.int32)
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    #norm difference
    error = (np.linalg.norm(img_watermarked_stacked-img_stacked))/(np.linalg.norm(img_stacked))
    return error

def perceptibility_jain(img, watermark, scale):
    #watermarked image
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    #stacking watermarked image
    img_watermarked = img_watermarked.astype(np.int32)
    img_watermarked_rows, img_watermarked_columns = img_watermarked.shape[:2] 
    img_watermarked_stacked = img_watermarked.reshape(img_watermarked_rows, -1)
    #stacking image
    img = img.astype(np.int32)
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    #norm difference
    error = (np.linalg.norm(img_watermarked_stacked-img_stacked))/(np.linalg.norm(img_stacked))
    return error

def perceptibility_jain_mod(img, watermark, scale):
    #watermarked image
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    #stacking watermarked image
    img_watermarked = img_watermarked.astype(np.int32)
    img_watermarked_rows, img_watermarked_columns = img_watermarked.shape[:2] 
    img_watermarked_stacked = img_watermarked.reshape(img_watermarked_rows, -1)
    #stacking image
    img = img.astype(np.int32)
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    #norm difference
    error = (np.linalg.norm(img_watermarked_stacked-img_stacked))/(np.linalg.norm(img_stacked))
    return error


def watermarkedplot(img,watermark,plottype):
    scales = np.arange(0.05,2.05,0.05)
    differences = []
    #liu tan
    if plottype == 1:
        for scale in scales:
            print(scale)
            difference = perceptibility_liutan(img, watermark, scale)
            differences.append(difference)
    #jain
    if plottype == 2:
        for scale in scales:
            print(scale)
            difference = perceptibility_jain(img, watermark, scale)
            differences.append(difference)
            
    #jain mod
    if plottype == 3:
        for scale in scales:
            print(scale)
            difference = perceptibility_jain_mod(img, watermark, scale)
            differences.append(difference)
    
    drawgraph_difference(scales,differences,plottype)
    
def drawgraph_difference(x,y,plottype):
    plt.plot(x,y,marker='o')
    plt.xlabel('Alpha')
    plt.ylabel('Error')
    #plt.show()
    #liutan
    if plottype == 1:
        plt.savefig('../out/watermarking/plots/perceptibility/liutan/perceptibility_liutan.png')
    if plottype == 2:
        plt.savefig('../out/watermarking/plots/perceptibility/jain/perceptibility_jain.png')
    if plottype == 3:
        plt.savefig('../out/watermarking/plots/perceptibility/jainmod/perceptibility_jain_mod.png')
    plt.show() 
    
    
    
    
    
    
    
    
    
#lowrank extraction error
    

def lowrank_image_liutan(img, watermark, scale, rank, save):
    #watermarked image
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)
    
    #extracting watermark using original extraction key and compressed watermarked image
    watermark_extracted = it.extract_watermark(img_watermarked_approx, watermarked_u, mat_s, watermarked_vh,
            scale=scale)
    watermark_extracted = reversepad(watermark_extracted, watermark)
    watermark_extracted = watermark_extracted.astype(np.int32)
    
    if save=='no':
        return watermark_extracted
    elif save=='yes':
        it.save_image(watermark_extracted,'../out/watermarking/robustness/lowrankextraction/liutan/extraction_rank_{}_alpha_{}.png'.format(rank,scale))
        

def lowrank_watermarked_image_liutan(img, watermark, scale, rank, save):
    #watermarked image
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)
    
    img_watermarked_approx = img_watermarked_approx.astype(np.int32)
    if save=='no':
        return img_watermarked_approx
    elif save=='yes':
        it.save_image(img_watermarked_approx,'../out/watermarking/robustness/lowrankembedding/liutan/embedding_rank_{}_alpha_{}.png'.format(rank,scale))
    
def lowrank_image_jain(img, watermark, scale, rank, save):
    #watermarked image
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)

    #extracting watermark using original extraction key and compressed watermarked image
    watermark_extracted = it.extract_watermark_jain(img_watermarked_approx, img, watermark_vh, scale)
    watermark_extracted = reversepad(watermark_extracted, watermark)
    watermark_extracted = watermark_extracted.astype(np.int32)
    
    if save=='no':
        return watermark_extracted
    elif save=='yes':
        it.save_image(watermark_extracted,'../out/watermarking/robustness/lowrankextraction/jain/extraction_rank_{}_alpha_{}.png'.format(rank,scale))
    
    
def lowrank_watermarked_image_jain(img, watermark, scale, rank, save):
    #watermarked image
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)
    img_watermarked_approx = img_watermarked_approx.astype(np.int32)
    if save=='no':
        return img_watermarked_approx
    elif save=='yes':
        it.save_image(img_watermarked_approx,'../out/watermarking/robustness/lowrankembedding/jain/embedding_rank_{}_alpha_{}.png'.format(rank,scale))
    
    
def lowrank_image_jain_mod(img, watermark, scale, rank,save):
    #watermarked image
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)
    #extracting watermark using original extraction key and compressed watermarked image
    watermark_extracted = it.extract_watermark_jain_mod(img_watermarked, img, watermark_vh, scale=scale)
    watermark_extracted = reversepad(watermark_extracted, watermark)
    watermark_extracted = watermark_extracted.astype(np.int32)
    if save=='no':
        return watermark_extracted
    elif save=='yes':
        it.save_image(watermark_extracted,'../out/watermarking/robustness/lowrankextraction/jainmod/extraction_rank_{}_alpha_{}.png'.format(rank,scale))
    
def lowrank_watermarked_image_jain_mod(img, watermark, scale, rank,save):
    #watermarked image
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)
    img_watermarked_approx = img_watermarked_approx.astype(np.int32)
    if save=='no':
        return img_watermarked_approx
    elif save=='yes':
        it.save_image(img_watermarked_approx,'../out/watermarking/robustness/lowrankembedding/jainmod/embedding_rank_{}_alpha_{}.png'.format(rank,scale))
    
    
def lowrank_error_liutan(img, watermark, scale, rank):
    #watermarked image
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)
    #extracting watermark using original extraction key and compressed watermarked image
    watermark_extracted = it.extract_watermark(img_watermarked_approx, watermarked_u, mat_s, watermarked_vh,
            scale=scale)
    watermark_extracted = reversepad(watermark_extracted, watermark)
    #stacking extracted watermark
    watermark_extracted = watermark_extracted.astype(np.float64)
    watermark_extracted_rows, watermark_extracted_columns = watermark_extracted.shape[:2] 
    watermark_extracted_stacked = watermark_extracted.reshape(watermark_extracted_rows, -1)
    #stacking original watermark
    watermark = watermark.astype(np.float64)
    watermark_rows, watermark_columns = watermark.shape[:2] 
    watermark_stacked = watermark.reshape(watermark_rows, -1)
    #norm difference
    error = (np.linalg.norm(watermark_extracted_stacked-watermark_stacked))/(np.linalg.norm(watermark_stacked))
    return error

def lowrank_error_jain(img, watermark, scale, rank):
    #watermarked image
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)
    #extracting watermark using original extraction key and compressed watermarked image
    watermark_extracted = it.extract_watermark_jain(img_watermarked_approx, img, watermark_vh, scale)
    watermark_extracted = reversepad(watermark_extracted, watermark)
    #stacking extracted watermark
    watermark_extracted = watermark_extracted.astype(np.float64)
    watermark_extracted_rows, watermark_extracted_columns = watermark_extracted.shape[:2] 
    watermark_extracted_stacked = watermark_extracted.reshape(watermark_extracted_rows, -1)
    #stacking original watermark
    watermark = watermark.astype(np.float64)
    watermark_rows, watermark_columns = watermark.shape[:2] 
    watermark_stacked = watermark.reshape(watermark_rows, -1)
    #norm difference
    error = (np.linalg.norm(watermark_extracted_stacked-watermark_stacked))/(np.linalg.norm(watermark_stacked))
    return error

def lowrank_error_jain_mod(img, watermark, scale, rank):
    #watermarked image
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)
    #extracting watermark using original extraction key and compressed watermarked image
    watermark_extracted = it.extract_watermark_jain_mod(img_watermarked_approx, img, watermark_vh, scale=scale)
    watermark_extracted = reversepad(watermark_extracted, watermark)
    #stacking extracted watermark
    watermark_extracted = watermark_extracted.astype(np.float64)
    watermark_extracted_rows, watermark_extracted_columns = watermark_extracted.shape[:2] 
    watermark_extracted_stacked = watermark_extracted.reshape(watermark_extracted_rows, -1)
    #stacking original watermark
    watermark = watermark.astype(np.float64)
    watermark_rows, watermark_columns = watermark.shape[:2] 
    watermark_stacked = watermark.reshape(watermark_rows, -1)
    #norm difference
    error = (np.linalg.norm(watermark_extracted_stacked-watermark_stacked))/(np.linalg.norm(watermark_stacked))
    return error

def lowrank_extractionerror_plot_liutan(img,watermark):
    alphas = (0.05,0.1,0.5,0.75)
    ranks = np.arange(1,300)
    errors0 = []
    for rank in ranks:
        error0 = lowrank_error_liutan(img,watermark,alphas[0],rank)
        errors0.append(error0)
        print("liutan",rank)
    errors1 = []
    for rank in ranks:
        error1 = lowrank_error_liutan(img,watermark,alphas[1],rank)
        errors1.append(error1)
        print("liutan",rank)
    errors2 = []
    for rank in ranks:
        error2 = lowrank_error_liutan(img,watermark,alphas[2],rank)
        errors2.append(error2)
        print("liutan",rank)
    errors3 = []
    for rank in ranks:
        error3 = lowrank_error_liutan(img,watermark,alphas[3],rank)
        errors3.append(error3)
        print("liutan",rank)
    plt.plot(errors0,label="a = {0}".format(alphas[0]))
    plt.plot(errors1,label="a = {0}".format(alphas[1]))
    plt.plot(errors2,label="a = {0}".format(alphas[2]))
    plt.plot(errors3,label="a = {0}".format(alphas[3]))
    plt.xlabel('Rank')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('../out/watermarking/plots/lowrankcompression/liutan/lowrank_extractionerror_liutan.eps',bbox_inches='tight')


def lowrank_extractionerror_plot_jain(img,watermark):
    alphas = (0.05,0.1,0.5,0.75)
    ranks = np.arange(1,300)
    errors0 = []
    for rank in ranks:
        error0 = lowrank_error_jain(img,watermark,alphas[0],rank)
        errors0.append(error0)
        print("jain",rank)
    errors1 = []
    for rank in ranks:
        error1 = lowrank_error_jain(img,watermark,alphas[1],rank)
        errors1.append(error1)
        print("jain",rank)
    errors2 = []
    for rank in ranks:
        error2 = lowrank_error_jain(img,watermark,alphas[2],rank)
        errors2.append(error2)
        print("jain",rank)
    errors3 = []
    for rank in ranks:
        error3 = lowrank_error_jain(img,watermark,alphas[3],rank)
        errors3.append(error3)
        print("jain",rank)
    plt.plot(errors0,label="a = {0}".format(alphas[0]))
    plt.plot(errors1,label="a = {0}".format(alphas[1]))
    plt.plot(errors2,label="a = {0}".format(alphas[2]))
    plt.plot(errors3,label="a = {0}".format(alphas[3]))
    plt.xlabel('Rank')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('../out/watermarking/plots/lowrankcompression/jain/lowrank_extractionerror_jain.eps',bbox_inches='tight')
    
def lowrank_extractionerror_plot_jain_mod(img,watermark):
    alphas = (0.05,0.1,0.25,0.5,0.75)
    ranks = np.arange(1,300)
    errors0 = []
    for rank in ranks:
        error0 = lowrank_error_jain_mod(img,watermark,alphas[0],rank)
        errors0.append(error0)
        print("jain mod",rank)
    errors1 = []
    for rank in ranks:
        error1 = lowrank_error_jain_mod(img,watermark,alphas[1],rank)
        errors1.append(error1)
        print("jain mod",rank)
    errors2 = []
    for rank in ranks:
        error2 = lowrank_error_jain_mod(img,watermark,alphas[2],rank)
        errors2.append(error2)
        print("jain mod",rank)
    errors3 = []
    for rank in ranks:
        error3 = lowrank_error_jain_mod(img,watermark,alphas[3],rank)
        errors3.append(error3)
        print("jain mod",rank)
    plt.plot(errors0,label="a = {0}".format(alphas[0]))
    plt.plot(errors1,label="a = {0}".format(alphas[1]))
    plt.plot(errors2,label="a = {0}".format(alphas[2]))
    plt.plot(errors3,label="a = {0}".format(alphas[3]))
    plt.xlabel('Rank')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('../out/watermarking/plots/lowrankcompression/jainmod/lowrank_extractionerror_jain_mod.eps',bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    
#cropping tests 
    
def crop_left(img, number):
    img = img[:,number:,:]
    return img

def crop_right(img, number):
    img = img[:,:-number,:]
    return img

def crop_bottom(img, number):
    img = img[:-number,:,:]
    return img

def crop_top(img, number):
    img = img[number:,:,:]
    return img

def crop_image_liutan(img, watermark, scale, number, side):
    
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    #img_stacked = img_stacked.astype(np.int32)
    if side == 'left':
        cropped_watermarked_image = crop_left(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)
    elif side == 'right':
        cropped_watermarked_image = crop_right(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)     
    elif side == 'bottom':
        cropped_watermarked_image = crop_bottom(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)      
    elif side == 'top':
        cropped_watermarked_image = crop_top(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)
        
    it.save_image(cropped_watermarked_image_padded,'../out/watermarking/cropping/embedding/liutan/embedding_alpha_{}_cropped_{}_from_{}.png'.format(scale, number,side))
        
def crop_extract_watermark_liutan(img, watermark, scale, number, side):
    
    img_watermarked, watermarked_u, mat_s, watermarked_vh = it.embed_watermark(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    #img_stacked = img_stacked.astype(np.int32)
    if side == 'left':
        cropped_watermarked_image = crop_left(img_watermarked, number)
    elif side == 'right':
        cropped_watermarked_image = crop_right(img_watermarked, number)
    elif side == 'bottom':
        cropped_watermarked_image = crop_bottom(img_watermarked, number)
    elif side == 'top':
        cropped_watermarked_image = crop_top(img_watermarked, number)
        
    cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
    cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)
    watermark_extracted = it.extract_watermark(cropped_watermarked_image_padded, watermarked_u, mat_s, watermarked_vh,
            scale=scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    watermark_extracted_final = watermark_extracted_final.astype(np.int32)
    it.save_image(watermark_extracted_final,'../out/watermarking/cropping/extracting/liutan/extrated_watermark_alpha_{}_cropped_{}_from_{}.png'.format(scale, number,side))

def crop_image_jain(img, watermark, scale, number, side):
    
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    #img_stacked = img_stacked.astype(np.int32)
    if side == 'left':
        cropped_watermarked_image = crop_left(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)
    elif side == 'right':
        cropped_watermarked_image = crop_right(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)     
    elif side == 'bottom':
        cropped_watermarked_image = crop_bottom(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)      
    elif side == 'top':
        cropped_watermarked_image = crop_top(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)
        
    it.save_image(cropped_watermarked_image_padded,'../out/watermarking/cropping/embedding/jain/embedding_alpha_{}_cropped_{}_from_{}.png'.format(scale, number,side))
 
def crop_extract_watermark_jain(img, watermark, scale, number, side):
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    #img_stacked = img_stacked.astype(np.int32)
    if side == 'left':
        cropped_watermarked_image = crop_left(img_watermarked, number)
    elif side == 'right':
        cropped_watermarked_image = crop_right(img_watermarked, number)
    elif side == 'bottom':
        cropped_watermarked_image = crop_bottom(img_watermarked, number)
    elif side == 'top':
        cropped_watermarked_image = crop_top(img_watermarked, number)
        
    cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
    cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)
    watermark_extracted = it.extract_watermark_jain(cropped_watermarked_image_padded, img, watermark_vh, scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    watermark_extracted_final = watermark_extracted_final.astype(np.int32)
    it.save_image(watermark_extracted_final,'../out/watermarking/cropping/extracting/jain/extrated_watermark_alpha_{}_cropped_{}_from_{}.png'.format(scale, number,side))

def crop_image_jain_mod(img, watermark, scale, number, side):
    
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    #img_stacked = img_stacked.astype(np.int32)
    if side == 'left':
        cropped_watermarked_image = crop_left(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)
    elif side == 'right':
        cropped_watermarked_image = crop_right(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)     
    elif side == 'bottom':
        cropped_watermarked_image = crop_bottom(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)      
    elif side == 'top':
        cropped_watermarked_image = crop_top(img_watermarked, number)
        cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
        cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)
        
    it.save_image(cropped_watermarked_image_padded,'../out/watermarking/cropping/embedding/jainmod/embedding_alpha_{}_cropped_{}_from_{}.png'.format(scale, number,side))

 
def crop_extract_watermark_jain_mod(img, watermark, scale, number, side):
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    #img_stacked = img_stacked.astype(np.int32)
    if side == 'left':
        croppedwatermarked_image = crop_left(img_watermarked, number)
    elif side == 'right':
        cropped_watermarked_image = crop_right(img_watermarked, number)
    elif side == 'bottom':
        cropped_watermarked_image = crop_bottom(img_watermarked, number)
    elif side == 'top':
        cropped_watermarked_image = crop_top(img_watermarked, number)
        
    cropped_watermarked_image = it.padimage3d(img, cropped_watermarked_image)
    cropped_watermarked_image_padded = cropped_watermarked_image.astype(np.int32)
    watermark_extracted = it.extract_watermark_jain_mod(cropped_watermarked_image_padded, img, watermark_vh, scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    watermark_extracted_final = watermark_extracted_final.astype(np.int32)
    it.save_image( watermark_extracted_final,'../out/watermarking/cropping/extracting/jainmod/extrated_watermark_alpha_{}_cropped_{}_from_{}.png'.format(scale, number,side))

