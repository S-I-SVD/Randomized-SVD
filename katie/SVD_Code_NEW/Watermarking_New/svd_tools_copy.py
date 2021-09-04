#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:22:45 2021

@author: katie
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from timeit import timeit



'''
Computes the randomized singular value decomposition of the input matrix.
'''
def randomized_svd(matrix, rank, oversample=0, power_iterations = 0, full_matrices = False):
    rows, columns = matrix.shape

    # Create a random projection matrix
    projector = np.random.rand(columns, rank + oversample)

    # Sample from the column space of X
    sample = matrix @ projector

    # Perform power iteration
    for i in range(0, power_iterations):
        sample = matrix @ (matrix.T @ sample)

    orthogonal, r = np.linalg.qr(sample)

    # Project X into the sampled subspace
    projected = orthogonal.T @ matrix

    # Obtain the SVD for this smaller matrix and recover the SVD for matrix
    u_projected, s, v = np.linalg.svd(projected, full_matrices)
    u = orthogonal @ u_projected

    return u, s, v

def compressed_svd(matrix, rank, oversample=5, density=3):
    rows, cols = matrix.shape

    if density == None:
        test_matrix = np.random.rand(rank + oversample, rows)
    else:
        # Sparse random test matrix
        test_matrix = np.random.choice(a=[1, 0, -1], 
                p=[1/(2*density), 1 - 1/density, 1/(2*density)],
                size=(rank + oversample, rows))

    # Y
    sketch = test_matrix @ matrix

    # Outer product of sketch matrix with itself to obtain right singular vectors
    # B
    outer = sketch @ sketch.T

    # Ensure symmetry
    #outer = 1/2 * (outer + outer.T) 

    #T, V
    outer_eigenvalues, outer_eigenvectors = la.eigh(outer)
    outer_eigenvectors_trunc = np.flip(outer_eigenvectors[:, -rank:],axis=1)
    outer_eigenvalues_trunc = np.flip(outer_eigenvalues[-rank:])
    print(outer_eigenvalues)
    print(outer_eigenvalues_trunc)

    singular_values = np.sqrt(outer_eigenvalues_trunc)
    singular_values_matrix = np.diag(singular_values)

    right_singular_vectors = sketch.T @ outer_eigenvectors_trunc @ la.inv(singular_values_matrix)
    scaled_left_singular_vectors = matrix @ right_singular_vectors

    left_singular_vectors_updated, singular_values_updated, right_sv_multiplier = la.svd(scaled_left_singular_vectors, full_matrices=False)

    right_singular_vectors_updated = right_singular_vectors @ right_sv_multiplier.T

    return left_singular_vectors_updated, singular_values_updated, right_singular_vectors_updated.T

'''
Project a matrix onto the given SVD components
'''
def svd_project(mat, components, centering='s', rank=None):
    u, s, vh = centered_svd(mat, full_matrices=False, centering=centering, rank=rank)
    return u[:, components] @ np.diag(s[components]) @ vh[components, :]

'''
Computes a rank <rank> approximation of a matrix with optional oversampling and randomization 
'''
def rank_k_approx(x, rank=None, ratio=None, min_energy=None, mode='deterministic', 
        oversample=0, power_iterations=0, centering='s'):

    if rank == None and ratio == None and min_energy == None:
        raise Exception('Must provide either rank, ratio, or min_energy')

    if rank == None and ratio != None:
        rank = int(la.matrix_rank(x) * ratio)

    u, s, vh = centered_svd(x, centering=centering, full_matrices=False, mode=mode,
            rank=rank, oversample=oversample, power_iterations=power_iterations)

    if centering in [None, 's', 'simple']:
        means_mat = 0
    elif centering in ['c', 'col', 'cols', 'column', 'columns']:
        column_means = np.array([np.mean(x[:, i]) for i in range(0, x.shape[1])])
        means_mat = column_means * np.ones(x.shape)
    elif centering in ['r', 'row', 'rows']:
        row_means = np.array([np.mean(x[i, :]) for i in range(0, x.shape[0])])
        means_mat = (row_means * np.ones(x.T.shape)).T
    elif centering in ['d', 'double', 'both']:
        column_means = np.array([np.mean(x[:, i]) for i in range(0, x.shape[1])])
        row_means = np.array([np.mean(x[i, :]) for i in range(0, x.shape[0])])
        x_mean = np.mean(x)

        column_means_mat = column_means * np.ones(x.shape)
        row_means_mat = (row_means * np.ones(x.T.shape)).T
        means_mat = row_means_mat + column_means_mat - x_mean


    '''
    if (not randomized):
        u,s,vh = np.linalg.svd(mat, full_matrices=False)
    else:
        u,s,vh = randomized_svd(mat, rank=rank, oversample=oversample, 
                power_iterations=power_iterations, full_matrices=False)
    '''

    if rank == None:
        if ratio == None:
            s_cumsum = [sum(s[0:i]) for i in range(0,s.size)] / sum(s)
            rank = int(np.arange(1,s.size+1)[s_cumsum >= min_energy][0])
        else:
            rank = int(la.matrix_rank(x) * ratio) 

    return u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank, :] + means_mat

'''
Compresses a video using a low rank approximation. Takes in either the rank of the approximation or the compression ratio.
'''
def compress_video(video, ratio=None, rank=None, mode='deterministic', oversample=0):
    video_shape = video.shape
    num_frames = video_shape[0]

    video_flattened = video.reshape(num_frames, -1)
    #print('video_flattened.rank = %d' % la.matrix_rank(video_flattened))
    #print('video_flattened.shape = %s' % (video_flattened.shape,))
    video_flattened_approx = compress_image(video_flattened, 
            ratio=ratio, rank=rank, mode=mode, oversample=oversample)
    #print('video_flattened_approx.shape = %s' % (video_flattened_approx.shape,))
    #print('video_flattened_approx.rank = %d' % la.matrix_rank(video_flattened_approx))

    video_approx = video_flattened_approx.reshape(video_shape)

    return video_approx

def map_sv(mat, fun):
    U, S, V = la.svd(mat, full_matrices=False)
    SS = np.array([fun(s) for s in S])
    return U @ np.diag(SS) @ V

def replace_singular_vectors(source, destination, indices, side):
    us, ss, vs = la.svd(source, full_matrices = False) 
    ud, sd, vd = la.svd(destination, full_matrices = False) 

    if side == 'left':
        for i in indices:
            ud[:,i] = us[:,i]
    elif side == 'right':
        for i in indices:
            vd[i, :] = vs[i,:]

    print('replace svec: %s . %s . %s' % (ud.shape, sd.size, vd.shape))
    return ud @ np.diag(sd) @ vd

'''
Compute SVD of a matrix with row, column, or double centering for PCA
'''
def centered_svd(x, centering=None, full_matrices=False, mode='deterministic', oversample=10, rank=None,
        power_iterations=0):
    centering = centering.lower()

    if centering in [None, 's', 'simple']:
        x_centered = x
    elif centering in ['c', 'col', 'cols', 'column', 'columns']:
        column_means = np.array([np.mean(x[:, i]) for i in range(0, x.shape[1])])
        column_means_mat = column_means * np.ones(x.shape)
        x_centered = x - column_means_mat
    elif centering in ['r', 'row', 'rows']:
        row_means = np.array([np.mean(x[i, :]) for i in range(0, x.shape[0])])
        row_means_mat = (row_means * np.ones(x.T.shape)).T
        x_centered = x - row_means_mat
    elif centering in ['d', 'double', 'both']:
        column_means = np.array([np.mean(x[:, i]) for i in range(0, x.shape[1])])
        row_means = np.array([np.mean(x[i, :]) for i in range(0, x.shape[0])])
        x_mean = np.mean(x)

        column_means_mat = column_means * np.ones(x.shape)
        row_means_mat = (row_means * np.ones(x.T.shape)).T
        double_means_mat = row_means_mat + column_means_mat - x_mean

        x_centered = x - double_means_mat

    mode_letter = mode[0].lower()
    if mode_letter == 'd':
        # deterministic
        return la.svd(x_centered, full_matrices=full_matrices)
    elif mode_letter == 'r':
        # randomized
        return randomized_svd(x_centered, rank=rank, oversample=oversample, full_matrices=full_matrices)
    elif mode_letter == 'c':
        return compressed_svd(x_centered, rank=rank, oversample=oversample)


def sv_plot(mat, fname=None, show=False, log=False):
    fig, ax = plt.subplots()
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    if log:
        ax.plot(np.arange(s.size)+1, np.log(1+s), 'ro')
    else:
        ax.plot(np.arange(s.size)+1, s, 'ro')
    ax.set_ylabel(r'$\log(1+\sigma_n)$')
    ax.set_xlabel('n')
    if fname != None:
        fig.set_size_inches(3, 3)
        fig.tight_layout()
        fig.savefig(fname, bbox_inches='tight')

    if show:
        ax.show()
    
'''
Displays a cumulative sum plot of the singular values of the input matrix
'''
def sv_cumsum_plot(mat, fname=None, show=False):
    fig, ax = plt.subplots()
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    ax.plot(1 + np.arange(s.size), [sum(s[0:i]) for i in range(0,s.size)] / sum(s), 'ro')
    ax.set_ylabel('Cumulative Sum Proportion')
    ax.set_xlabel('Number of Singular Values')
    if fname != None:
        fig.set_size_inches(3, 3)
        fig.tight_layout()
        fig.savefig(fname, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

'''
Display the generalized scree plot (Zhang) 
'''
def scree_plot(ax, mat, num_sv, title='Scree Plot', scale=''):
    num_sv = num_sv+1
    ranks = np.arange(1,num_sv)

    if scale == 'log':
        mat = np.log(mat - np.min(mat) + 1)

    def residual_proportion(x, rank, centering):
        x_approx = rank_k_approx(x, rank=rank, centering=centering)
        return (la.norm(x - x_approx) / la.norm(x))

    residual_proportions = dict( [ ( s, [residual_proportion(mat, rank, s) for rank in ranks]) for s in ['s', 'r', 'c', 'd']] )
    
    ax.plot(ranks, residual_proportions['s'], color='black', marker='o')
    #ax.plot(ranks - 1/2, residual_proportions['r'], color='red', marker='o')
    #ax.plot(ranks - 1/2, residual_proportions['c'], color='blue', marker='o')
    ax.plot(ranks, residual_proportions['r'], color='red', marker='o')
    ax.plot(ranks, residual_proportions['c'], color='blue', marker='o')
    ax.plot(ranks, residual_proportions['d'], color='purple', marker='o')

    ax.legend(['SSVD', 'RSVD', 'CSVD', 'DSVD'], loc='upper right')

    ax.set_title(title)
    ax.set_xlabel('No. SVD Components')
    ax.set_ylabel('Residual Proportion')

    ax.set_xticks(range(1, num_sv))

    #plt.show()

'''
a = np.random.rand(100, 100)
scree_plot(a, 10)
'''

def modify_sigmas_add(img, scalar):
    img_type = img.dtype
    img = img.astype(np.float64)

        
    # Stack color channels
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    
    U, S, VT = la.svd(img_stacked, full_matrices=False)

    length_S = len(S)
    for i in range(length_S):
        S[i] = S[i]  + scalar
    S = np.diag(S)
    
    reconstruction = U @ S @ VT
    reconstruction = reconstruction.reshape(img_rows,img_columns,-1)
    
    return reconstruction
