import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2
import math

import svd_tools as svdt
import watermark as wm

from PIL import Image

import png


'''
Compresses an image using a low rank approximation. Takes in either the rank of the approximation or the compression ratio.
'''
def compress_image(img, ratio=None, rank=None, min_energy=None, mode='deterministic', oversample=0, power_iterations=0):
    return process_image((lambda x: svdt.rank_k_approx(x, rank=rank, ratio=ratio,
        oversample=oversample, min_energy=min_energy, 
        mode=mode, power_iterations=power_iterations)), img)


def embed_watermark(img, watermark, scale=1):
    '''
    Embed a watermark in a multi-channel image using the Liu & Tan watermarking scheme.
    Corresponds to Algorithm 2.2 in the paper.
    Args:
        img: The image in which to embed the watemark (A in Algorithm 2.2)
        watermark: The watermark to embed in the image (W in Algorithm 2.2)
        scale: Scaling factor (alpha in Algorithm 2.2)
    Returns: (A_W, U_W, S, (V_W)^T) (corresponding to the symbols in Algorithm 2.2)
    '''
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

    # Handle overflow/underflow issues
    '''
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
    #watermark_type = watermark.dtype
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

    # Handle overflow/underflow issues
    '''
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

def get_jain_term(img, watermark, mod=False):
    img = img.astype(np.float64)
    watermark = watermark.astype(np.float64)
    watermark = pad_image(watermark, img.shape)

    # Stack color channels
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    watermark_rows, watermark_columns = watermark.shape[:2] 
    watermark_stacked = watermark.reshape(watermark_rows, -1)

    u, s, vh = la.svd(img_stacked)
    uw, sw, vwh = la.svd(watermark_stacked)
    num_sv = len(sw)

    sw_mat = np.pad(np.diag(sw), [(0, watermark_stacked.shape[0] - num_sv), (0, watermark_stacked.shape[1] - num_sv)])
    print([x.shape for x in [u, uw, sw_mat, vh]])
    if mod:
        jain_term = (uw @ sw_mat @ vh).reshape(img_rows, img_columns, -1)
    else:
        jain_term = (u @ uw @ sw_mat @ vh).reshape(img_rows, img_columns, -1)

    return jain_term



'''
Extract a watermark from an image using the Jain algorithm
'''
def extract_watermark_jain(img_watermarked, img_original, watermark_vh, scale, size=None):
    #watermark_type = watermark_vh.dtype
    #img_type = img_watermarked.dtype
        
    # Stack color channels
    img_rows, img_columns = img_watermarked.shape[:2] 
    img_watermarked = img_watermarked.astype(np.float64)
    img_watermarked_stacked = img_watermarked.reshape(img_rows, -1)

    orig_rows, orig_columns = img_original.shape[:2] 
    img_original = img_original.astype(np.float64)
    img_original_stacked = img_original.reshape(orig_rows, -1)

    #watermark_vh = watermark_vh.astype(np.float64)
    
    # Extract watermark
    watermark_stacked = wm.extract_watermark_jain(img_watermarked_stacked, img_original_stacked, watermark_vh, scale=scale)
    watermark = watermark_stacked.reshape(img_rows, img_columns, -1)

    # Get rid of redundant dimensions
    if watermark.shape[2] == 1:
        watermark.shape = watermark.shape[:2]

    # Handle overflow/underflow issues
    '''
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
    #watermark_type = watermark.dtype
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

    # Handle overflow/underflow issues
    '''
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
    #watermark_type = watermark_vh.dtype
    #img_type = img_watermarked.dtype
        
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

    # Handle overflow/underflow issues
    '''
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



'''
Apply a function to a colored image, automatically handling dtype and 
stacking (for colored images) issues
'''
def process_image(fun, img):
    img_type = img.dtype
    img = img.astype(np.float64)
        
    # Stack color channels
    rows, columns = img.shape[:2] 
    img_stacked = img.reshape(rows, -1)
    img_rank = np.linalg.matrix_rank(img_stacked)
    
    # Compute rank <rank> approximation of image
    fun_img_stacked = fun(img_stacked)
    fun_img = fun_img_stacked.reshape(rows, columns, -1)

    # Get rid of redundant dimensions
    if fun_img.shape[2] == 1:
        fun_img.shape = fun_img.shape[:2]

    # Handle overflow/underflow issues
    if np.issubdtype(img_type, np.integer):
        fun_img = np.clip(fun_img, 0, 255)
    else:
        fun_img = np.clip(fun_img, 0, 1)

    fun_img = fun_img.astype(img_type)
    return fun_img

'''
Pad an image in the first two dimensions (width x height)
'''
def pad_image(img, new_shape):
    img_padded =  np.pad(img, [(0, new_shape[0] - img.shape[0]), (0, new_shape[1] - img.shape[1])] + [(0,0) for i in range(len(img.shape)) if i > 1])
    return img_padded

'''
Compute the peak signal to noise ration (PSNR) of two images
'''
def psnr(img1, img2):
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mean_square_error = np.mean((img1 - img2) ** 2)
    print(mean_square_error)
    if mean_square_error == 0:
        # Same image
        return 100
    return 10 * math.log10(1. / math.sqrt(mean_square_error))

def load_image(path, dtype=np.uint8):
    return np.asarray(Image.open(path)).astype(dtype)

def save_image(img, path):
    Image.fromarray(img.clip(0,255).astype(np.uint8)).convert('RGB').save(path)

raccoon = load_image('res/public/raccoon.jpg')
fox = load_image('res/public/fox.jpg')
husky = load_image('res/public/husky.jpg')
noise = load_image('out/images/noise.jpg')
checker = load_image('res/images/checker.jpg')
checker_noise = load_image('res/images/checker_noise.jpg')
