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

'''
Project a matrix onto the given SVD components
'''
def svd_project(mat, components, centering='s', rank=None):
    u, s, vh = centered_svd(mat, full_matrices=False, centering='s', rank=rank)
    return u[:, components] @ np.diag(s[components]) @ vh[components, :]

'''
Computes a rank <rank> approximation of a matrix with optional oversampling and randomization 
'''
def rank_k_approx(x, rank=None, ratio=None, min_energy=None, randomized=False, 
        oversample=0, power_iterations=0, centering='s'):

    if rank == None and ratio == None and min_energy == None:
        raise Exception('Must provide either rank, ratio, or min_energy')

    if rank == None and ratio != None:
        rank = int(la.matrix_rank(x) * ratio)

    u, s, vh = centered_svd(x, centering=centering, full_matrices=False, randomized=randomized,
            oversample=oversample, power_iterations=power_iterations)

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
Compresses an image using a low rank approximation. Takes in either the rank of the approximation or the compression ratio.
'''
def compress_image(img, ratio=None, rank=None, min_energy=None, randomized=False, oversample=0, power_iterations=0):
    img_type = img.dtype
        
    # Stack color channels
    rows, columns = img.shape[:2] 
    img_stacked = img.reshape(rows, -1)
    img_rank = np.linalg.matrix_rank(img_stacked)
    
    # Compute rank <rank> approximation of image
    img_approx_stacked = rank_k_approx(img_stacked, rank=rank, ratio=ratio,
            oversample=oversample, min_energy=min_energy,
            randomized=randomized, power_iterations=power_iterations)
    img_approx = img_approx_stacked.reshape(rows, columns, -1)

    # Get rid of redundant dimensions
    if img_approx.shape[2] == 1:
        img_approx.shape = img_approx.shape[:2]

    # Handle overflow/underflow issues
    if np.issubdtype(img_type, np.integer):
        img_approx = np.clip(img_approx, 0, 255)
    else:
        img_approx = np.clip(img_approx, 0, 1)

    return img_approx.astype(img_type)

'''
Compresses a video using a low rank approximation. Takes in either the rank of the approximation or the compression ratio.
'''
def compress_video(video, ratio=None, rank=None, randomized=False, oversample=0):
    video_shape = video.shape
    num_frames = video_shape[0]

    video_flattened = video.reshape(num_frames, -1)
    #print('video_flattened.rank = %d' % la.matrix_rank(video_flattened))
    #print('video_flattened.shape = %s' % (video_flattened.shape,))
    video_flattened_approx = compress_image(video_flattened, 
            ratio=ratio, rank=rank, randomized=randomized, oversample=oversample)
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
def centered_svd(x, centering=None, full_matrices=False, randomized=False, oversample=10, rank=None,
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

    if not randomized:
        return la.svd(x_centered, full_matrices=full_matrices)
    else:
        return randomized_svd(x_centered, rank=rank, oversample=oversample, full_matrices=full_matrices)



'''
Displays a log-plot of the singular values of the input matrix
'''
def svd_log_plot(mat):
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    plt.plot(np.arange(s.size)+1,np.log(s)/np.log(10), 'ro')
    plt.show()

'''
Displays a cumulative sum plot of the singular values of the input matrix
'''
def svd_cumsum_plot(mat):
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    plt.plot(1 + np.arange(s.size), [sum(s[0:i]) for i in range(0,s.size)] / sum(s), 'ro')
    plt.show()

'''
Display the generalized scree plot (Zhang) 
'''
def scree_plot(mat, num_sv, title='Scree Plot'):
    ranks = np.arange(2,num_sv)

    def residual_proportion(x, rank, centering):
        x_approx = rank_k_approx(x, rank=rank, centering=centering)
        return (la.norm(x - x_approx) / la.norm(x))

    residual_proportions = dict( [ ( s, [residual_proportion(mat, rank, s) for rank in ranks]) for s in ['s', 'r', 'c', 'd']] )
    
    plt.plot(ranks, residual_proportions['s'], color='black', marker='o')
    plt.plot(ranks - 1/2, residual_proportions['r'], color='red', marker='o')
    plt.plot(ranks - 1/2, residual_proportions['c'], color='blue', marker='o')
    plt.plot(ranks, residual_proportions['d'], color='purple', marker='o')

    plt.legend(['SSVD', 'RSVD', 'CSVD', 'DSVD'], loc='upper right')

    plt.title(title)
    plt.xlabel('No. SVD Components')
    plt.ylabel('Residual Proportion')

    plt.show()

'''
a = np.random.rand(100, 100)
scree_plot(a, 10)
'''

