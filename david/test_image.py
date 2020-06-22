import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from timeit import timeit

from svd_tools import *

# Time randomized vs deterministic SVD on images
def time_svd():
    for rank in [5,10,25,50,100]:
        print('Randomized SVD (rank %d): %.2f seconds' %
                ( rank,
                  timeit(lambda : rank_k_approx(img, rank=rank, randomized=True), number=100)
                )
             )
        print('SVD (rank %d): %.2f seconds' %
                ( rank,
                  timeit(lambda : rank_k_approx(img, rank=rank, randomized=False), number=100)
                )
             )
        print('---')

def display_svd(img):
    randomized = True

    # Load a 2x3 grid of subplots
    f, ax = plt.subplots(4, 3)

    ratios = np.array([[.5, .25, .1], [.05, .025, .01], [.5, .25, .1], [.05, .025, .01]])
    titles = np.empty_like(ratios, 'S')
            
    for t in [(i,j) for i in range(0,2) for j in range(0,3)]:
        # Show images in subplots
        ax[t].imshow(compress_image(img, ratio=ratios[t], randomized=randomized, oversample=0))

        # Apply subplot titles and turn off axes
        ax[t].set_title('%.1f%%' % (ratios[t] * 100))
        ax[t].axis('off')

    for t in [(i,j) for i in range(2,4) for j in range(0,3)]:
        # Show images in subplots
        ax[t].imshow(compress_image(img, ratio=ratios[t], randomized=randomized, oversample=10))

        # Apply subplot titles and turn off axes
        ax[t].set_title('%.1f%%' % (ratios[t] * 100))
        ax[t].axis('off')

    # Show plot
    plt.show()

'''
Plot the logarithms of the singular values for a matrix
'''
def svd_log_plot(mat):
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    plt.plot(np.arange(s.size)+1,np.log(s)/np.log(10), 'ro')
    plt.show()

'''
Plot n vs the sum of the first n sigular values of a matrix
'''
def svd_cumsum_plot(mat):
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    plt.plot(1 + np.arange(s.size), [sum(s[0:i]) for i in range(0,s.size)] / sum(s), 'ro')
    plt.show()

display_svd(np.asarray(Image.open('raccoon.jpg')))
#time_them()
#svd_cumsum_plot(img_stacked)

