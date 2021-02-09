#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:04:40 2021

@author: katie
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import svd_tools as svdt
import scipy as sp
import scipy.sparse

from PIL import Image

'''
Embed a watermark in  using the Liu & Tan algorithm
'''

def embed_watermark(mat, watermark, scale=1):
    mat_rows, mat_columns = mat.shape
    watermark_rows, watermark_columns = watermark.shape

    if mat_rows < watermark_rows or mat_columns < watermark_columns:
        print('Watermark must be smaller than matrix')
        return

    mat_u, mat_s, mat_vh = la.svd(mat)
    mat_num_sv = len(mat_s)
    
    # Compute the rectangular "diagonal" singular value matrix
    mat_s_matrix = np.pad(np.diag(mat_s), 
            [(0, mat_rows - mat_num_sv), (0, mat_columns - mat_num_sv)])
    watermark_padded = np.pad(watermark, 
            [(0, mat_rows - watermark_rows), (0, mat_columns - watermark_columns)])
    mat_s_matrix_watermarked = mat_s_matrix + scale * watermark_padded

    watermarked_u, watermarked_s, watermarked_vh = la.svd(mat_s_matrix_watermarked)
    watermarked_num_sv = len(watermarked_s)
    watermarked_s_matrix = np.pad(np.diag(watermarked_s), 
            [(0, mat_rows - watermarked_num_sv), (0, mat_columns - watermarked_num_sv)])

    mat_watermarked = mat_u @ watermarked_s_matrix @ mat_vh

    return mat_watermarked, watermarked_u, mat_s_matrix, watermarked_vh

def embed_watermark_jain(mat, watermark, scale=1, term=False):
    mat_rows, mat_columns = mat.shape
    watermark_rows, watermark_columns = watermark.shape

    if mat_rows < watermark_rows or mat_columns < watermark_columns:
        print('Watermark must be smaller than matrix')
        return

    mat_u, mat_s, mat_vh = la.svd(mat)
    mat_num_sv = len(mat_s)
    
    # Compute the rectangular "diagonal" singular value matrix
    mat_s_matrix = np.pad(np.diag(mat_s), 
            [(0, mat_rows - mat_num_sv), (0, mat_columns - mat_num_sv)])

    # Pad watermark to match the sizes
    watermark_padded = np.pad(watermark, 
            [(0, mat_rows - watermark_rows), (0, mat_columns - watermark_columns)])

    watermark_u, watermark_s, watermark_vh = la.svd(watermark_padded)
    watermark_num_sv = len(watermark_s)
    watermark_rows, watermark_columns = mat.shape

    watermark_s_matrix = np.pad(np.diag(watermark_s), [(0, watermark_rows - watermark_num_sv), (0, watermark_columns - watermark_num_sv)])
    watermark_pcs = watermark_u @ watermark_s_matrix
    mat_s_matrix_watermarked = mat_s_matrix + scale * watermark_pcs

    mat_watermarked = mat_u @ mat_s_matrix_watermarked @ mat_vh
    jain_term = mat_u @ watermark_u @ watermark_s_matrix @ mat_vh

    if term:
        return mat_watermarked, watermark_vh, jain_term
    else:
        return mat_watermarked, watermark_vh

def extract_watermark_jain(mat_watermarked, mat_original, watermark_vh, scale):
    mat_u, mat_s, mat_vh = la.svd(mat_original)
    #watermark_pcs = (mat_u.conj().T @ (mat_watermarked - mat_original) @ mat_vh.conj().T) / scale
    watermark_pcs = (la.inv(mat_u) @ (mat_watermarked - mat_original) @ la.inv(mat_vh)) / scale
    return watermark_pcs @ watermark_vh



def embed_watermark_jain_mod(mat, watermark, scale=1, term=False):
    mat_rows, mat_columns = mat.shape
    watermark_rows, watermark_columns = watermark.shape

    if mat_rows < watermark_rows or mat_columns < watermark_columns:
        print('Watermark must be smaller than matrix')
        return

    mat_u, mat_s, mat_vh = la.svd(mat)
    mat_num_sv = len(mat_s)
    
    # Compute the rectangular "diagonal" singular value matrix
    mat_s_matrix = np.pad(np.diag(mat_s), 
            [(0, mat_rows - mat_num_sv), (0, mat_columns - mat_num_sv)])

    # Pad watermark to match the sizes
    watermark_padded = np.pad(watermark, 
            [(0, mat_rows - watermark_rows), (0, mat_columns - watermark_columns)])

    watermark_u, watermark_s, watermark_vh = la.svd(watermark_padded)
    watermark_num_sv = len(watermark_s)
    watermark_rows, watermark_columns = mat.shape

    watermark_s_matrix = np.pad(np.diag(watermark_s), [(0, watermark_rows - watermark_num_sv), (0, watermark_columns - watermark_num_sv)])

    mat_watermarked = mat + scale * watermark_u @ watermark_s_matrix @ mat_vh
    jain_mod_term = watermark_u @ watermark_s_matrix @ mat_vh

    if term:
        return mat_watermarked, watermark_vh, jain_mod_term
    else:
        return mat_watermarked, watermark_vh

    
    return mat_watermarked, watermark_vh

def extract_watermark_jain_mod(mat_watermarked, mat_original, watermark_vh, scale):
    mat_u, mat_s, mat_vh = la.svd(mat_original)
    return (mat_watermarked - mat_original) @ la.inv(mat_vh) @ watermark_vh / scale


def extract_watermark(mat_watermarked, watermarked_u, mat_s_matrix, watermarked_vh, scale):
    _, watermarked_s, _ = la.svd(mat_watermarked)
    mat_watermarked_rows, mat_watermarked_cols = mat_watermarked.shape
    num_sv = len(watermarked_s)
    watermarked_s_matrix = np.pad(np.diag(watermarked_s), 
            [(0, mat_watermarked_rows - num_sv), (0, mat_watermarked_cols - num_sv)])
    return (watermarked_u @ watermarked_s_matrix @ watermarked_vh - mat_s_matrix) / scale
