import numpy as np
import matplotlib.pyplot as plt

from timeit import timeit

'''
Computes the randomized singular value decomposition of the input matrix.
'''
def randomized_svd(matrix, rank, oversample=0, full_matrices = False):
    rows, columns = matrix.shape

    # Create a random projection matrix
    projector = np.random.rand(columns, rank + oversample)

    # Sample from the column space of X
    sample = matrix @ projector

    orthogonal, r = np.linalg.qr(sample)

    # Project X into the sampled subspace
    projected = orthogonal.T @ matrix

    # Obtain the SVD for this smaller matrix and recover the SVD for matrix
    u_projected, s, v = np.linalg.svd(projected, full_matrices)
    u = orthogonal @ u_projected

    return u, s, v

'''
Computes a rank <rank> approximation of a matrix with optional oversampling and randomization 
'''
def rank_k_approx(mat, rank, randomized=False, oversample=0):
    if (not randomized):
        u,s,vh = np.linalg.svd(mat, full_matrices=False)
    else:
        u,s,vh = randomized_svd(mat, rank=rank, oversample=oversample, full_matrices=False)

    return u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank, :]

'''
Compresses an image using a low rank approximation. Takes in either the rank of the approximation or the compression ratio.
'''
def compress_image(img, ratio=None, rank=None, randomized=False, oversample=0):
    img_type = img.dtype
        
    # Stack color channels
    rows, columns = img.shape[:2] 
    img_stacked = img.reshape(-1, columns)
    img_rank = np.linalg.matrix_rank(img_stacked)

    if(rank == None):
        if(ratio == None):
            raise Exception('compress_image must be passed either rank or ratio')
        rank = int(ratio * img_rank)

    # Compute rank <rank> approximation of image
    img_approx_stacked = rank_k_approx(img_stacked, rank=rank, 
            oversample=oversample, randomized=randomized)
    img_approx = img_approx_stacked.reshape(rows, columns, -1)

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

    video_flattened = video.reshape(-1, num_frames)
    video_flattened_approx = compress_image(video_flattened, 
            ratio=ratio, rank=rank, randomized=randomized, oversample=oversample)

    video_approx = video_flattened_approx.reshape(video_shape)

    return video_approx

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
