import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import skvideo as vid
import skvideo.io
import skvideo.utils

from timeit import timeit

# Load image
img = np.asarray(Image.open('raccoon.jpg'))
# img.shape = (r, c, 3)

img_stacked = img.reshape(-1, img.shape[1])
img_flattened = img.reshape(-1)
#img_stacked = img.reshape(r * 3, c)


img_rank = np.linalg.matrix_rank(img.reshape(-1, img.shape[0]))
print('rank is %d' % img_rank)

# Load video
video = vid.utils.rgb2gray(vid.io.vread('school_smol.mp4'))
video_shape = video.shape
num_frames = video_shape[0]
print('video loaded %s' % (video.shape,))

# (num_frames, rows, cols, num_channels)
video_flattened = video.reshape(-1, num_frames)
print('video flattened %s' % (video_flattened.shape,))
plt.imshow(video_flattened)
plt.axis('off')
plt.show()

def randomized_svd(matrix, rank, oversample, full_matrices = False):
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

def rank_n_approx(img, rank=None, randomized=False, color=False, oversample=0):
    img_type = img.dtype
    img = img.astype('float32')
    
    # Stack color channels
    rows, columns = img.shape[:2] 
    img_stacked = img.reshape(-1, columns)

    # Compute rank <rank> approximation of image
    if (not randomized):
        u, s, vh = np.linalg.svd(img_stacked, full_matrices=False)
    else:
        u, s, vh = randomized_svd(img_stacked, rank=rank, oversample=oversample, full_matrices=False)

    # Compute rank <rank> approximation of img
    img_approx_stacked = u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank, :]

    # Normalize values to within valid range
    def normalize_img(img):
        M = np.max(img_approx_stacked)
        m = np.min(img_approx_stacked)
        old_range = M - m
        if np.issubdtype(img_type, np.integer):
            # Image dtype is integral, so original range is 0-255
            old_max = 255
        else:
            # Image dtype is floating, so original range is 0-1
            old_max = 1

        new_range = min(M/old_max, 1) - max(m/old_max, 0)

        return (img - m) / old_range * new_range + max(m/old_max, 0) 

    #img_approx_stacked_normalized = normalize_img(img_approx_stacked)
    #img_approx_normalized = img_approx_stacked_normalized.reshape(rows, columns,-1)
    img_approx = img_approx_stacked.reshape(rows, columns, -1) / 255

    return img_approx


# Time functions
def time_them():
    for rank in [5,10,25,50,100]:
        print('Randomized SVD (rank %d): %.2f seconds' %
                ( rank,
                  timeit(lambda : rank_n_approx(img, rank=rank, randomized=True), number=100)
                )
             )
        print('SVD (rank %d): %.2f seconds' %
                ( rank,
                  timeit(lambda : rank_n_approx(img, rank=rank, randomized=False), number=100)
                )
             )
        print('---')

def display_them():
    randomized = True

    # Load a 2x3 grid of subplots
    f, ax = plt.subplots(4, 3)

    ratios = np.array([[.5, .25, .1], [.05, .025, .01], [.5, .25, .1], [.05, .025, .01]])
    titles = np.empty_like(ratios, 'S')
            
    for t in [(i,j) for i in range(0,2) for j in range(0,3)]:
        # Show images in subplots
        ax[t].imshow(rank_n_approx(img, int(img_rank * ratios[t]), randomized, oversample=0))

        # Apply subplot titles and turn off axes
        ax[t].set_title('%.1f%%' % (ratios[t] * 100))
        ax[t].axis('off')

    for t in [(i,j) for i in range(2,4) for j in range(0,3)]:
        # Show images in subplots
        ax[t].imshow(rank_n_approx(img, int(img_rank * ratios[t]), randomized, oversample=10))

        # Apply subplot titles and turn off axes
        ax[t].set_title('%.1f%%' % (ratios[t] * 100))
        ax[t].axis('off')

    # Show plot
    plt.show()

def svd_log_plot(mat):
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    plt.plot(np.arange(s.size)+1,np.log(s)/np.log(10), 'ro')
    plt.show()

def svd_cumsum_plot(mat):
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    plt.plot(1 + np.arange(s.size), [sum(s[0:i]) for i in range(0,s.size)] / sum(s), 'ro')
    plt.show()

#display_them()
#time_them()
#svd_cumsum_plot(img_stacked)

