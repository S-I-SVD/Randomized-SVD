import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from timeit import timeit

from svd_tools import *

# Time randomized vs deterministic SVD on images
def time_svd(mat):
    print('Matrix size = %s' % (mat.shape,))
    print('Matrix rank = %d' % la.matrix_rank(mat))
    oversample = 10
    for rank in [5,10,25,50,100]:
        print('Randomized SVD (rank %d + %d oversampling): %.2f seconds' %
                ( rank,
                  oversample,
                  timeit(lambda : rank_k_approx(mat, rank=rank, randomized=True, 
                      oversample=oversample), 
                      number=100)
                )
             )
        print('SVD (rank %d): %.2f seconds' %
                ( rank,
                  timeit(lambda : rank_k_approx(mat, rank=rank, randomized=False), number=100)
                )
             )
        print('---')

def display_svd_oversampling(img):
    randomized = True

    # Load a 2x3 grid of subplots
    f, ax = plt.subplots(4, 3)

    #ratios = np.array([[.5, .25, .1], [.05, .025, .01], [.5, .25, .1], [.05, .025, .01]])
    #energies = np.array([[.99, .9, .8], [.7, .6, .5]])
    ranks = np.array([[100,50,30],[20,10,5],[100,50,30],[20,10,5]])
    titles = np.empty_like(ranks, 'S')
            
    for t in [(i,j) for i in range(0,2) for j in range(0,3)]:
        # Show images in subplots
        ax[t].imshow(compress_image(img, rank=ranks[t], randomized=randomized, oversample=10))

        # Apply subplot titles and turn off axes
        ax[t].set_title('Rank %d (o.s. 10)' % ranks[t])
        ax[t].axis('off')

    for t in [(i,j) for i in range(2,4) for j in range(0,3)]:
        # Show images in subplots
        ax[t].imshow(compress_image(img, rank=ranks[t], randomized=randomized))

        # Apply subplot titles and turn off axes
        ax[t].set_title('Rank %d' % ranks[t])
        ax[t].axis('off')

    # Show plot
    plt.show()

def display_svd_randomized(img):
    # Load a 2x3 grid of subplots
    f, ax = plt.subplots(4, 3)

    #ratios = np.array([[.5, .25, .1], [.05, .025, .01], [.5, .25, .1], [.05, .025, .01]])
    #energies = np.array([[.99, .9, .8], [.7, .6, .5]])
    ranks = np.array([[100,50,30],[20,10,5],[100,50,30],[20,10,5]])
    titles = np.empty_like(ranks, 'S')
            
    for t in [(i,j) for i in range(0,2) for j in range(0,3)]:
        # Show images in subplots
        ax[t].imshow(compress_image(img, rank=ranks[t], randomized=True, oversample=10))

        # Apply subplot titles and turn off axes
        ax[t].set_title('Rank %d (o.s. 10)' % ranks[t])
        ax[t].axis('off')

    for t in [(i,j) for i in range(2,4) for j in range(0,3)]:
        # Show images in subplots
        ax[t].imshow(compress_image(img, rank=ranks[t], randomized=False))

        # Apply subplot titles and turn off axes
        ax[t].set_title('Rank %d' % ranks[t])
        ax[t].axis('off')

    # Show plot
    plt.show()


def energy_vs_accuracy(image, energies):
    image_flattened = image.reshape(img.shape[0], -1)
    compressed_images = [compress_image(img, min_energy=energy) for energy in energies]
    compressed_images_flattened = [compressed_image.reshape(compressed_image.shape[0],-1) for compressed_image in compressed_images]
    distances = np.array([la.norm(image_flattened - compressed_image) for compressed_image in compressed_images_flattened])
    plt.plot(energies, distances/10000, 'ro')
    plt.xlabel('Minimum Energy')
    yl = plt.ylabel(r'$\dfrac{\Vert X - X_e \Vert_F}{10000}$', labelpad=30)
    yl.set_rotation(0)
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

'''
Plot n vs the sum of the first n sigular values of a matrix
'''
def svd_frobenius_cumsum_plot(mat):
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    plt.plot(1 + np.arange(s.size), [np.sqrt(sum(s[0:i]**2)) for i in range(0,s.size)] / np.sqrt(sum(s**2)), 'ro')
    plt.show()

img = np.asarray(Image.open('res/raccoon.jpg').convert('L'))
#energy_vs_accuracy(img, [.8 + k*0.01 for k in range(0,20)])
#time_svd(img.reshape(img.shape[0],-1))
#display_svd(np.asarray(Image.open('res/raccoon.jpg')))
#display_svd_randomized(img)
#svd_log_plot(img.reshape(img.shape[0], -1))
svd_frobenius_cumsum_plot(img.reshape(img.shape[0],-1))


#time_them()
#svd_cumsum_plot(img_stacked)

