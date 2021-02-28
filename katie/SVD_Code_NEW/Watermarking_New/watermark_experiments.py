import matplotlib.pyplot as plt
import matplotlib
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
    img_watermarked = img_watermarked.astype(np.int32)
    if save=='no':
        return img_watermarked
    elif save=='yes':
        matplotlib.image.imsave('../out/watermarking/watermarked_image/liutan/watermarked_image_alpha_{}.png'.format(scale), img_watermarked)
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
        matplotlib.image.imsave('../out/watermarking/extracted_watermark/liutan/extracted_watermark_alpha_{}.png'.format(scale), watermark_extracted_final)
    
def watermark_embed_jain(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jain
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    if save=='no':
        return img_watermarked
    elif save=='yes':
        matplotlib.image.imsave('../out/watermarking/watermarked_image/jain/watermarked_image_alpha_{}.png'.format(scale), img_watermarked)
    
def watermark_extract_jain(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jain
    img_watermarked, watermark_vh = it.embed_watermark_jain(img, watermark, scale=scale)
    watermark_extracted = it.extract_watermark_jain(img_watermarked, img, watermark_vh, scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    watermark_extracted_final = watermark_extracted_final.astype(np.int32)
    if save=='no':
        return watermark_extracted_final
    elif save=='yes':
        matplotlib.image.imsave('../out/watermarking/extracted_watermark/jain/extracted_watermark_alpha_{}.png'.format(scale), watermark_extracted_final)
    
def watermark_embed_jain_mod(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jainmod
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    if save=='no':
        return img_watermarked
    elif save=='yes':
        matplotlib.image.imsave('../out/watermarking/watermarked_image/jainmod/watermarked_image_alpha_{}.png'.format(scale), img_watermarked)
    
def watermark_extract_jain_mod(img, watermark, scale, save):
    #embeds watermark into image. if save == 'yes', then it will save to out/watermarking/watermarked_image/jainmod
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    watermark_extracted = it.extract_watermark_jain_mod(img_watermarked, img, watermark_vh, scale)
    watermark_extracted_final = reversepad(watermark_extracted, watermark)
    watermark_extracted_final = watermark_extracted_final.astype(np.int32)
    if save=='no':
        return watermark_extracted_final
    elif save=='yes':
        matplotlib.image.imsave('../out/watermarking/extracted_watermark/jainmod/extracted_watermark_alpha_{}.png'.format(scale), watermark_extracted_final)
        
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
        matplotlib.image.imsave('../out/watermarking/robustness/lowrankextraction/liutan/extraction_rank_{}_alpha_{}.png'.format(rank,scale), watermark_extracted)
        

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
        matplotlib.image.imsave('../out/watermarking/robustness/lowrankembedding/liutan/embedding_rank_{}_alpha_{}.png'.format(rank,scale), img_watermarked_approx)
    
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
        matplotlib.image.imsave('../out/watermarking/robustness/lowrankextraction/jain/extraction_rank_{}_alpha_{}.png'.format(rank,scale), watermark_extracted)
    
    
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
        matplotlib.image.imsave('../out/watermarking/robustness/lowrankembedding/jain/embedding_rank_{}_alpha_{}.png'.format(rank,scale), img_watermarked_approx)
    
    
def lowrank_image_jain_mod(img, watermark, scale, rank,save):
    #watermarked image
    img_watermarked, watermark_vh = it.embed_watermark_jain_mod(img, watermark, scale=scale)
    img_watermarked = img_watermarked.astype(np.int32)
    #applying low rank compression to watermarked image
    img_watermarked_approx = it.lowrankapprox(img_watermarked,rank)
    #extracting watermark using original extraction key and compressed watermarked image
    watermark_extracted = it.extract_watermark_jain_mod(img_watermarked, img, watermark_vh, scale=scale)
    watermark_extracted = watermark_extracted.astype(np.int32)
    watermark_extracted = reversepad(watermark_extracted, watermark)
    if save=='no':
        return watermark_extracted
    elif save=='yes':
        matplotlib.image.imsave('../out/watermarking/robustness/lowrankextraction/jainmod/extraction_rank_{}_alpha_{}.png'.format(rank,scale), watermark_extracted)
    
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
        matplotlib.image.imsave('../out/watermarking/robustness/lowrankembedding/jainmod/embedding_rank_{}_alpha_{}.png'.format(rank,scale), img_watermarked_approx)
    
    
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
    

def lowrank_extractionerror_plot_liutan(img,watermark):
    alphas = (0.05,0.1,0.15,0.2,0.25,0.5,0.75)
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
    errors4 = []
    for rank in ranks:
        error4 = lowrank_error_liutan(img,watermark,alphas[4],rank)
        errors4.append(error4)
        print("liutan",rank)
    errors5 = []
    for rank in ranks:
        error5 = lowrank_error_liutan(img,watermark,alphas[5],rank)
        errors5.append(error5)
        print("liutan",rank)
    errors6 = []
    for rank in ranks:
        error6 = lowrank_error_liutan(img,watermark,alphas[6],rank)
        errors6.append(error6)
        print("liutan",rank)
    plt.plot(errors0,label="a = {0}".format(alphas[0]))
    plt.plot(errors1,label="a = {0}".format(alphas[1]))
    plt.plot(errors2,label="a = {0}".format(alphas[2]))
    plt.plot(errors3,label="a = {0}".format(alphas[3]))
    plt.plot(errors4,label="a = {0}".format(alphas[4]))
    plt.plot(errors5,label="a = {0}".format(alphas[5]))
    plt.plot(errors6,label="a = {0}".format(alphas[6]))
    plt.xlabel('Rank')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('../out/watermarking/plots/lowrankcompression/liutan/lowrank_extractionerror_liutan.png')


def lowrank_extractionerror_plot_jain(img,watermark):
    alphas = (0.05,0.1,0.15,0.2,0.25,0.5,0.75)
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
    errors4 = []
    for rank in ranks:
        error4 = lowrank_error_jain(img,watermark,alphas[4],rank)
        errors4.append(error4)
        print("jain",rank)
    errors5 = []
    for rank in ranks:
        error5 = lowrank_error_jain(img,watermark,alphas[5],rank)
        errors5.append(error5)
        print("jain",rank)
    errors6 = []
    for rank in ranks:
        error6 = lowrank_error_jain(img,watermark,alphas[6],rank)
        errors6.append(error6)
        print("jain",rank)
    plt.plot(errors0,label="a = {0}".format(alphas[0]))
    plt.plot(errors1,label="a = {0}".format(alphas[1]))
    plt.plot(errors2,label="a = {0}".format(alphas[2]))
    plt.plot(errors3,label="a = {0}".format(alphas[3]))
    plt.plot(errors4,label="a = {0}".format(alphas[4]))
    plt.plot(errors5,label="a = {0}".format(alphas[5]))
    plt.plot(errors6,label="a = {0}".format(alphas[6]))
    plt.xlabel('Rank')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('../out/watermarking/plots/lowrankcompression/jain/lowrank_extractionerror_jain.png')
    
def lowrank_extractionerror_plot_jain_mod(img,watermark):
    alphas = (0.05,0.1,0.15,0.2,0.25,0.5,0.75)
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
    errors4 = []
    for rank in ranks:
        error4 = lowrank_error_jain_mod(img,watermark,alphas[4],rank)
        errors4.append(error4)
        print("jain mod",rank)
    errors5 = []
    for rank in ranks:
        error5 = lowrank_error_jain_mod(img,watermark,alphas[5],rank)
        errors5.append(error5)
        print("jain mod",rank)
    errors6 = []
    for rank in ranks:
        error6 = lowrank_error_jain_mod(img,watermark,alphas[6],rank)
        errors6.append(error6)
    plt.plot(errors0,label="a = {0}".format(alphas[0]))
    plt.plot(errors1,label="a = {0}".format(alphas[1]))
    plt.plot(errors2,label="a = {0}".format(alphas[2]))
    plt.plot(errors3,label="a = {0}".format(alphas[3]))
    plt.plot(errors4,label="a = {0}".format(alphas[4]))
    plt.plot(errors5,label="a = {0}".format(alphas[5]))
    plt.plot(errors6,label="a = {0}".format(alphas[6]))
    plt.xlabel('Rank')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('../out/watermarking/plots/lowrankcompression/jainmod/lowrank_extractionerror_jain_mod.png')
    
##lowrank_extractionerror_plot_liutan(view,tree)
##lowrank_extractionerror_plot_jain(view,tree)
#lowrank_extractionerror_plot_jain_mod(view,tree)