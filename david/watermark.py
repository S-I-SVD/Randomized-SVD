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
def embed_watermark_liutan(mat, watermark, scale=1):
    '''
    Embed a watermark in a matrix using the Liu & Tan watermarking scheme.
    Corresponds to Algorithm 2.2 in the paper.
    Args:
        img: The matrix in which to embed the watemark (A in Algorithm 2.2)
        watermark: The watermark to embed in the matrix (W in Algorithm 2.2)
        scale: Scaling factor (alpha in Algorithm 2.2)
    Returns: (A_W, U_W, S, (V_W)^T) (corresponding to the symbols in Algorithm 2.2)
    '''
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
embed_watermark = embed_watermark_liutan

def extract_watermark_liutan(mat_watermarked, watermarked_u, mat_s_matrix, watermarked_vh, scale):
    '''
    Extact a watermark from a matrix using the Liu & Tan watermarking scheme.
    Corresponds to Algorithm 2.3 in the paper.
    Args:
        mat_watermarked: The watermarked matrix (\widetilde{A}_W in Algorithm 2.3)
        watermarked_u: U_W from Algorithm 2.3 (watermarked_u from the return statement of the function embed_watermark_liutan)
        mat_s_matrix: Singular value matrix of the original unwatermarked matrix (S from Algorithm 2.2/2.3, mat_s_matrix from the function embed_watermark_liutan)
        watermarked_vh: (V_W)^T from Algorithm 2.3 (watermarked_vh from the return statement of embed_watermark_liutan)
        scale:
        scale: Scaling factor (alpha in Algorithm 2.2/2.3)
    Returns: The recovered image (\widetilde{W} in Algorithm 2.3)
    '''
    _, watermarked_s, _ = la.svd(mat_watermarked)
    mat_watermarked_rows, mat_watermarked_cols = mat_watermarked.shape
    num_sv = len(watermarked_s)
    watermarked_s_matrix = np.pad(np.diag(watermarked_s), 
            [(0, mat_watermarked_rows - num_sv), (0, mat_watermarked_cols - num_sv)])
    return (watermarked_u @ watermarked_s_matrix @ watermarked_vh - mat_s_matrix) / scale
extract_watermark = extract_watermark_liutan

def embed_watermark_jain(mat, watermark, scale=1, term=False):
    '''
    Embed a watermark in a matrix using the Jain et al watermarking scheme.
    Corresponds to Algorithm 2.4 in the paper.
    Args:
        img: The matrix in which to embed the watemark (A in Algorithm 2.4)
        watermark: The watermark to embed in the matrix (W in Algorithm 2.4)
        scale: Scaling factor (alpha in Algorithm 2.4)
        term: Whether or not to also return the "Jain term" (the term added to A in Equation 3.1) in addition to the watermarked matrix and the key.
    Returns: (A_W, (V_W)^T, [Jain term as above if term==True]) (corresponding to the symbols in Algorithm 2.4)
    '''
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
    '''
    Extact a watermark from a matrix using the Jain watermarking scheme.
    Corresponds to Algorithm 2.5 in the paper.
    Args:
        mat_watermarked: The watermarked matrix (\widetilde{A}_W in Algorithm 2.5)
        mat_original: The original, unwatermarked matrix (A in Algorithm 2.4/2.5)
        watermark_vh: (V_W)^T from Algorithm 2.4/2.5 (watermark_vh from the return statement of embed_watermark_jain)
        scale: Scaling factor (alpha in Algorithm 2.4/2.5)
    Returns: The recovered image (\widetilde{W} in Algorithm 2.5)
    '''
    mat_u, mat_s, mat_vh = la.svd(mat_original)
    #watermark_pcs = (mat_u.conj().T @ (mat_watermarked - mat_original) @ mat_vh.conj().T) / scale
    watermark_pcs = (la.inv(mat_u) @ (mat_watermarked - mat_original) @ la.inv(mat_vh)) / scale
    return watermark_pcs @ watermark_vh



def embed_watermark_jain_mod(mat, watermark, scale=1, term=False):
    '''
    Embed a watermark in a matrix using the proposed modified Jain et al watermarking scheme.
    Corresponds to Algorithm 3.1 in the paper.
    Args:
        img: The matrix in which to embed the watemark (A in Algorithm 3.1)
        watermark: The watermark to embed in the matrix (W in Algorithm 3.1)
        scale: Scaling factor (alpha in Algorithm 3.1)
        term: Whether or not to also return the "Jain mod term" (the term added to A in Step 3 of Algorithm 3.1) in addition to the watermarked matrix and the key.
    Returns: (A_W, (V_W)^T, [Jain mod term as above if term==True]) (corresponding to the symbols in Algorithm 3.1)
    '''
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
    '''
    Extact a watermark from a matrix using the proposed modified Jain watermarking scheme.
    Corresponds to Algorithm 3.1 in the paper.
    Args:
        mat_watermarked: The watermarked matrix (\widetilde{A}_W in Algorithm 3.1)
        mat_original: The original, unwatermarked matrix (A in Algorithm 3.1/3.2)
        watermark_vh: (V_W)^T from Algorithm 3.1/3.2 (watermark_vh from the return statement of embed_watermark_jain_mod)
        scale: Scaling factor (alpha in Algorithm 3.1/3.2)
    Returns: The recovered image (\widetilde{W} in Algorithm 3.2)
    '''
    mat_u, mat_s, mat_vh = la.svd(mat_original)
    return (mat_watermarked - mat_original) @ la.inv(mat_vh) @ watermark_vh / scale


