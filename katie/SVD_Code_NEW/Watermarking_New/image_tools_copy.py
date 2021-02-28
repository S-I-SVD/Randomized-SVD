#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 20:40:41 2021

@author: katie
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import svd_tools_copy as svdt
import scipy as sp
import scipy.sparse
import svd_tools_copy as svdt
import watermark_copy as wm
from PIL import Image
import png


'''
Compresses an image using a low rank approximation. Takes in either the rank of the approximation or the compression ratio.
'''
def compress_image(img, ratio=None, rank=None, min_energy=None, mode='deterministic', oversample=0, power_iterations=0):
    return process_image((lambda x: svdt.rank_k_approx(x, rank=rank, ratio=ratio,
        oversample=oversample, min_energy=min_energy, 
        mode=mode, power_iterations=power_iterations)), img)


'''
Embed a watermark in an image using the Liu & Tan algorithm
'''
def embed_watermark(img, watermark, scale=1):

    img_type = img.dtype
    img = img.astype(np.float64)
    watermark = watermark.astype(np.float64)

        
    # Stack color channels
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    watermark_rows, watermark_columns = watermark.shape[:2] 
    watermark_stacked = watermark.reshape(watermark_rows, -1)
    
    # Embed watermark in image
    img_watermarked_stacked, watermarked_u, mat_s, watermarked_vh = wm.embed_watermark(img_stacked, watermark_stacked, scale)
    img_watermarked = img_watermarked_stacked.reshape(img_rows, img_columns, -1)

    # Get rid of redundant dimensions
    if img_watermarked.shape[2] == 1:
        img_watermarked.shape = img_watermarked.shape[:2]

    # Handle overflow/underflow issues
    '''
    if np.issubdtype(img_type, np.integer):
        img_watermarked = np.clip(img_watermarked, 0, 255)
    else:
        img_watermarked = np.clip(img_watermarked, 0, 1)
    '''
    #img_watermarked = img_watermarked.astype(img_type)
    return img_watermarked, watermarked_u, mat_s, watermarked_vh
'''
Extract a watermark from an image using the Liu & Tan algorithm
'''
def extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh, scale, 
        mode='deterministic', rank=None, size=None):
    img_type = img_watermarked.dtype
    img_watermarked = img_watermarked.astype(np.float64)
        
    # Stack color channels
    img_rows, img_columns = img_watermarked.shape[:2] 
    img_watermarked_stacked = img_watermarked.reshape(img_rows, -1)
    
    # Extract watermark
    mode_letter = mode[0].lower()
    if mode_letter == 'd':
        # Deterministic
        _, watermarked_s, _ = la.svd(img_watermarked_stacked)
    elif mode_letter == 'r':
        # Randomized
        _, watermarked_s, _ = svdt.randomized_svd(img_watermarked_stacked, rank=rank)
    elif mode_letter == 'c':
        # Compressed
        _, watermarked_s, _ = svdt.compressed_svd(img_watermarked_stacked, rank=rank)

    mat_watermarked_rows, mat_watermarked_cols = img_watermarked_stacked.shape
    num_sv = len(watermarked_s)
    watermarked_s_matrix = np.pad(np.diag(watermarked_s), 
            [(0, mat_watermarked_rows - num_sv), (0, mat_watermarked_cols - num_sv)])

    watermark_stacked = (watermarked_u @ watermarked_s_matrix @ watermarked_vh - mat_s) / scale
    watermark = watermark_stacked.reshape(img_rows, img_columns, -1)

    # Get rid of redundant dimensions
    if watermark.shape[2] == 1:
        watermark.shape = watermark.shape[:2]
    '''
    # Handle overflow/underflow issues
    
    if np.issubdtype(img_type, np.integer):
        watermark = np.clip(watermark, 0, 255)
    else:
        watermark = np.clip(watermark, 0, 1)
    '''

    #watermark = watermark.astype(img_type)
    if size is None:
        return watermark
    else:
        return watermark[:size[0], :size[1]]


'''
Embed a watermark in an image using the Jain algorithm
'''
def embed_watermark_jain(img, watermark, scale=1, term=False):
    img_type = img.dtype
    watermark_type = watermark.dtype
    img = img.astype(np.float64)
    watermark = watermark.astype(np.float64)
    watermark = pad_image(watermark, img.shape)
        
    # Stack color channels
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    watermark_rows, watermark_columns = watermark.shape[:2] 
    watermark_stacked = watermark.reshape(watermark_rows, -1)
    
    # Embed watermark in image
    if term:
        img_watermarked_stacked, watermark_vh, jain_term_stacked = wm.embed_watermark_jain(img_stacked, watermark_stacked, scale, term=True)
        jain_term = jain_term_stacked.reshape(*img.shape)
    else:
        img_watermarked_stacked, watermark_vh = wm.embed_watermark_jain(img_stacked, watermark_stacked, scale, term=False)

    img_watermarked = img_watermarked_stacked.reshape(img_rows, img_columns, -1)

    # Get rid of redundant dimensions
    if img_watermarked.shape[2] == 1:
        img_watermarked.shape = img_watermarked.shape[:2]
    '''
    # Handle overflow/underflow issues
    if np.issubdtype(img_type, np.integer):
        img_watermarked = np.clip(img_watermarked, 0, 255)
    else:
        img_watermarked = np.clip(img_watermarked, 0, 1)
    '''
    #img_watermarked = img_watermarked.astype(img_type)
    #watermark_vh = watermark_vh.astype(watermark_type)
    
    if term:
        return img_watermarked, watermark_vh, jain_term
    else:
        return img_watermarked, watermark_vh

'''
Extract a watermark from an image using the Jain algorithm
'''
def extract_watermark_jain(img_watermarked, img_original, watermark_vh, scale, size=None):
    watermark_type = watermark_vh.dtype
    img_type = img_watermarked.dtype
    img_watermarked = img_watermarked.astype(np.float64)
    img_original = img_original.astype(np.float64)
        
    # Stack color channels
    img_rows, img_columns = img_watermarked.shape[:2] 
    img_watermarked_stacked = img_watermarked.reshape(img_rows, -1)

    orig_rows, orig_columns = img_original.shape[:2] 
    img_original_stacked = img_original.reshape(orig_rows, -1)

    watermark_vh = watermark_vh.astype(np.float64)
    
    # Extract watermark
    watermark_stacked = wm.extract_watermark_jain(img_watermarked_stacked, img_original_stacked, watermark_vh, scale=scale)
    watermark = watermark_stacked.reshape(img_rows, img_columns, -1)

    # Get rid of redundant dimensions
    if watermark.shape[2] == 1:
        watermark.shape = watermark.shape[:2]
    '''
    # Handle overflow/underflow issues
    if np.issubdtype(watermark_type, np.integer):
        watermark = np.clip(watermark, 0, 255)
    else:
        watermark = np.clip(watermark, 0, 1)
    '''
    #watermark = watermark.astype(watermark_type)

    if size == None:
        return watermark
    else:
        return watermark[:size[0], :size[1]]



def embed_watermark_jain_mod(img, watermark, scale=1, term=False):
    img_type = img.dtype
    watermark_type = watermark.dtype
    img = img.astype(np.float64)
    watermark = watermark.astype(np.float64)
    watermark = pad_image(watermark, img.shape)
        
    # Stack color channels
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    watermark_rows, watermark_columns = watermark.shape[:2] 
    watermark_stacked = watermark.reshape(watermark_rows, -1)
    
    # Embed watermark in image
    if term:
        img_watermarked_stacked, watermark_vh, jain_mod_term_stacked = wm.embed_watermark_jain_mod(img_stacked, watermark_stacked, scale, term=True)
        jain_mod_term = jain_mod_term_stacked.reshape(*img.shape)
    else:
        img_watermarked_stacked, watermark_vh = wm.embed_watermark_jain_mod(img_stacked, watermark_stacked, scale)

    img_watermarked = img_watermarked_stacked.reshape(img_rows, img_columns, -1)

    # Get rid of redundant dimensions
    if img_watermarked.shape[2] == 1:
        img_watermarked.shape = img_watermarked.shape[:2]
    '''
    # Handle overflow/underflow issues

    if np.issubdtype(img_type, np.integer):
        img_watermarked = np.clip(img_watermarked, 0, 255)
    else:
        img_watermarked = np.clip(img_watermarked, 0, 1)
    '''
    #img_watermarked = img_watermarked.astype(img_type)
    #watermark_vh = watermark_vh.astype(watermark_type)
    if term:
        return img_watermarked, watermark_vh, jain_mod_term
    else:
        return img_watermarked, watermark_vh
'''
Extract a watermark from an image using the Jain algorithm
'''
def extract_watermark_jain_mod(img_watermarked, img_original, watermark_vh, scale, size=None):
    watermark_type = watermark_vh.dtype
    img_type = img_watermarked.dtype
        
    # Stack color channels
    img_rows, img_columns = img_watermarked.shape[:2] 
    img_watermarked = img_watermarked.astype(np.float64)
    img_watermarked_stacked = img_watermarked.reshape(img_rows, -1)

    orig_rows, orig_columns = img_original.shape[:2] 
    img_original = img_original.astype(np.float64)
    img_original_stacked = img_original.reshape(orig_rows, -1)

    #watermark_vh = watermark_vh.astype(np.float64)
    
    # Extract watermark
    watermark_stacked = wm.extract_watermark_jain_mod(img_watermarked_stacked, img_original_stacked, watermark_vh, scale=scale)
    watermark = watermark_stacked.reshape(img_rows, img_columns, -1)

    # Get rid of redundant dimensions
    if watermark.shape[2] == 1:
        watermark.shape = watermark.shape[:2]
    '''
    # Handle overflow/underflow issues

    if np.issubdtype(watermark_type, np.integer):
        watermark = np.clip(watermark, 0, 255)
    else:
        watermark = np.clip(watermark, 0, 1)
    '''
    #watermark = watermark.astype(watermark_type)

    if size == None:
        return watermark
    else:
        return watermark[:size[0], :size[1]]
    
#additional watermarking tools

'''
Pad an image in the first two dimensions (width x height)
'''
def pad_image(img, new_shape):
    img_padded =  np.pad(img, [(0, new_shape[0] - img.shape[0]), (0, new_shape[1] - img.shape[1])] + [(0,0) for i in range(len(img.shape)) if i > 1])
    return img_padded

def load_image(path, dtype=np.uint8):
    return np.asarray(Image.open(path)).astype(dtype)

def save_image(img, path):
    Image.fromarray(img.clip(0,255).astype(np.uint8)).convert('RGB').save(path)
    
'''
Low rank approx (used in watermark_experiments)
'''
def lowrankapprox(img, k):
    #formatting
    img_type = img.dtype
    img = img.astype(np.float64)
    
    #stacking color channels
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    
    #computing svd
    U, S, VT = np.linalg.svd(img_stacked, full_matrices=False)
    S = np.diag(S)
    
    #construct approximate image from U, S, VT with rank k
    img_approx_stacked = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
    
    #reshaping
    img_approx = img_approx_stacked.reshape(img_rows, img_columns, -1)
    
    #get rid of redundant dimensions
    if img.shape[2]==1:
        img_approx.shape = img_approx.shape[:2]
    '''
    #handle overflow/underflow issues
    if np.issubdtype(img_type, np.integer):
        img_approx = np.clip(img_approx, 0, 255)
    else:
        img_approx = np.clip(img_approx, 0, 1)
    '''    
    #type
    #img_approx = img_approx.astype(img_type)
    #retun approximated image
    return img_approx