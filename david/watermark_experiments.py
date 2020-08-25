import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import svd_tools as svdt
import image_tools as imgt

from PIL import Image

import png

from watermark import *


def test_watermark_jain(scale=1):
    watermark_size = (5,5)
    mat = np.random.rand(10, 10)
    watermark = np.random.rand(*watermark_size)
    mat_watermarked, watermark_vh = embed_watermark_jain(mat, watermark, scale=scale)
    watermark_extracted = extract_watermark_jain(mat_watermarked, mat, watermark_vh,
            scale=scale)
    watermark_extracted = watermark_extracted[:watermark_size[0], :watermark_size[1]]
    print(la.norm(mat_watermarked - mat))
    print(la.norm(watermark - watermark_extracted))

def test_watermark_jain_round(scale=1):
    watermark_size = (7,7)
    mat = np.random.rand(10, 10) * 256
    watermark = np.random.rand(*watermark_size) * 256

    # mess it up
    mat_watermarked, watermark_vh = embed_watermark_jain(mat, watermark, scale=scale)
    print('a')
    print(mat_watermarked)
    mat_watermarked_2 = mat_watermarked.astype(np.uint8)
    mat_watermarked_2 = mat_watermarked_2.astype(np.float64)
    print('b')
    print(mat_watermarked_2)
    mat_watermarked = np.floor(mat_watermarked)
    print('c')
    print(mat_watermarked)
    print(watermark_vh.dtype)

    #watermark_vh = watermark_vh.astype(np.uint8)
    watermark_extracted = extract_watermark_jain(mat_watermarked, mat, watermark_vh,
            scale=scale)
    watermark_extracted = watermark_extracted[:watermark_size[0], :watermark_size[1]]
    print(la.norm(mat_watermarked - mat) / la.norm(mat))
    print(la.norm(watermark - watermark_extracted) / la.norm(watermark))

''' wip
def plot_watermark_perceptibility(scale=1, size=(128, 128)):
    mat = np.random.normal(scale=128, size=size)
    watermark = np.random.normal(scale=128, size=size)

    mat_watermarked_jain, vh_jain = embed_watermark_jain(mat, watermark, scale)
    mat_watermarked_jain_mod, vh_jain_mod = embed_watermark_jain_mod(mat, watermark, scale)
    mat_watermarked_liutan, u_liutan, s_liutan, vh_liutan = embed_watermark(mat, watermark, scale)
'''



def plot_watermark_perturb_errors_sparse(scale=1, min_mult=1, max_mult=20, num_points=20, 
        size=(128,128)):
    mults = np.linspace(min_mult, max_mult, num_points)
    relative_errors_jain = np.empty_like(mults)
    relative_errors_jain_mod = np.empty_like(mults)
    relative_errors_liutan = np.empty_like(mults)

    #mat = np.asarray(Image.open('res/images/raccoon.jpg').convert('L')).astype(np.float64)
    #mat = mat[:size[0], :size[1]]
    mat = np.random.normal(scale=128, size=size)
    watermark = np.random.normal(scale=128, size=size)

    # Embed watermark
    mat_watermarked_jain, vh_jain = embed_watermark_jain(mat, watermark, scale)
    mat_watermarked_jain_mod, vh_jain_mod = embed_watermark_jain_mod(mat, watermark, scale)
    mat_watermarked_liutan, u_liutan, s_liutan, vh_liutan = embed_watermark(mat, watermark, scale)

    # Perturb watermarked matrix
    for i in range(num_points):
        #perturbation = np.random.normal(scale=std_devs[i], size=size)
        #perturbation -= np.max(perturbation)
        perturbation = sp.sparse.rand(*size, density=0.05).A * mults[i]

        watermark_extracted_jain = extract_watermark_jain(mat_watermarked_jain + perturbation, 
                mat, vh_jain, scale)
        relative_errors_jain[i] = la.norm(watermark - watermark_extracted_jain) / la.norm(watermark)

        watermark_extracted_jain_mod = extract_watermark_jain_mod(mat_watermarked_jain_mod + perturbation, 
                mat, vh_jain_mod, scale)
        relative_errors_jain_mod[i] = la.norm(watermark - watermark_extracted_jain_mod) / la.norm(watermark)

        watermark_extracted_liutan = extract_watermark(mat_watermarked_liutan + perturbation, 
                u_liutan, s_liutan, vh_liutan, scale)
        relative_errors_liutan[i] = la.norm(watermark - watermark_extracted_liutan) / la.norm(watermark)


    fig, ax = plt.subplots()
    #ax.plot(mults, relative_errors_jain, label='Jain')
    ax.plot(mults, relative_errors_jain_mod, label='Jain Mod')
    ax.plot(mults, relative_errors_liutan, label='LiuTan')
    ax.legend()
    ax.set_title('Relative Errors in Watermark Extraction from Perturbed Image')
    ax.set_xlabel('Maximum Perturbation')
    ax.set_ylabel('Relative Error')

    # Save figure
    fig.set_size_inches(5,5)
    fig.savefig('out/watermark/watermark_sparse_perturb_errors.png', bbox_inches='tight', layout='landscape')
    plt.show()

def plot_watermark_perturb_errors_random(scale=1, min_stddev=.1, max_stddev=2, num_points=20, 
        size=(128,128)):
    std_devs = np.linspace(min_stddev, max_stddev, num_points)
    relative_errors_jain = np.empty_like(std_devs)
    relative_errors_jain_mod = np.empty_like(std_devs)
    relative_errors_liutan = np.empty_like(std_devs)
    norm_ps = np.empty_like(std_devs)

    #mat = np.asarray(Image.open('res/images/raccoon.jpg').convert('L')).astype(np.float64)
    #mat = mat[:size[0], :size[1]]
    mat = np.random.normal(scale=128, loc=128, size=size)
    watermark = np.random.normal(scale=128, loc=128, size=size)

    # Embed watermark
    mat_watermarked_jain, vh_jain = embed_watermark_jain(mat, watermark, scale)
    mat_watermarked_jain_mod, vh_jain_mod = embed_watermark_jain_mod(mat, watermark, scale)
    mat_watermarked_liutan, u_liutan, s_liutan, vh_liutan = embed_watermark(mat, watermark, scale)

    # Perturb watermarked matrix
    for i in range(num_points):
        perturbation = np.random.normal(scale=std_devs[i], size=size)

        watermark_extracted_jain = extract_watermark_jain(mat_watermarked_jain 
                + perturbation, mat, vh_jain, scale)
        relative_errors_jain[i] = la.norm(watermark 
                - watermark_extracted_jain) / la.norm(watermark)

        watermark_extracted_jain_mod = extract_watermark_jain_mod(
                mat_watermarked_jain_mod + perturbation, mat, vh_jain_mod, scale)
        relative_errors_jain_mod[i] = la.norm(watermark 
                - watermark_extracted_jain_mod) / la.norm(watermark)

        watermark_extracted_liutan = extract_watermark(mat_watermarked_liutan 
                + perturbation, u_liutan, s_liutan, vh_liutan, scale)
        relative_errors_liutan[i] = la.norm(watermark 
                - watermark_extracted_liutan) / la.norm(watermark)

        norm_ps[i] = la.norm(perturbation) / scale

    fig, ax = plt.subplots()
    ax.plot(std_devs, relative_errors_jain, label='Jain')
    ax.plot(std_devs, relative_errors_jain_mod, label='Jain Mod')
    ax.plot(std_devs, relative_errors_liutan, label='LiuTan')
    ax.plot(std_devs, norm_ps, label='norm P')
    ax.legend()
    ax.set_title('Relative Errors in Watermark Extraction from Perturbed Image')
    ax.set_xlabel('Standard Deviation of Perturbation Noise')
    ax.set_ylabel('Relative Error')

    # Save figure
    fig.set_size_inches(5,5)
    fig.savefig('out/watermark/watermark_random_perturb_errors.png', bbox_inches='tight', layout='landscape')

    plt.show()

def plot_watermark_perturb_errors_scales_random_jain(min_scale=.1, max_scale=1, num_scales=10, min_stddev=.1, max_stddev=2, num_stddevs=20, 
        size=(128,128)):
    std_devs = np.linspace(min_stddev, max_stddev, num_stddevs)
    scales = np.linspace(min_scale, max_scale, num_scales)
    relative_errors= np.empty((num_stddevs, num_scales))

    #mat = np.asarray(Image.open('res/images/raccoon.jpg').convert('L')).astype(np.float64)
    #mat = mat[:size[0], :size[1]]
    mat = np.random.normal(scale=128, loc=128, size=size)
    watermark = np.random.normal(scale=128, loc=128, size=size)


    # Perturb watermarked matrix
    for i in range(num_stddevs):
        for j in range(num_scales):
            std_dev = std_devs[i]
            scale = scales[j]

            # Embed watermark
            mat_watermarked, vh= embed_watermark_jain(mat, watermark, scale)
            
            # Perturb watermarked matrix
            perturbation = np.random.normal(scale=std_dev, size=size)
            mat_watermarked+= perturbation

            # Extract watermark and compute relative error
            watermark_extracted= extract_watermark_jain(mat_watermarked, 
                    mat, vh, scale)
            relative_errors[i,j] = la.norm(watermark - watermark_extracted) / la.norm(watermark)

    fig, ax = plt.subplots()
    for i in range(num_scales):
        ax.plot(std_devs, relative_errors[:, i], label=r'a=%.2f' % scales[i])
    ax.legend()
    ax.set_title('Relative Errors in Watermark Extraction from Perturbed Image (Jain)')
    ax.set_xlabel('Standard Deviation of Perturbation Noise')
    ax.set_ylabel('Relative Error')

    # Save figure
    fig.set_size_inches(5,5)
    #fig.savefig('out/watermark/watermark_random_perturb_errors.png', bbox_inches='tight', layout='landscape')

    fig.show()

def plot_watermark_perturb_errors_scales_random_jain_mod(min_scale=.1, max_scale=1, num_scales=10, min_stddev=.1, max_stddev=2, num_stddevs=20, 
        size=(128,128)):
    std_devs = np.linspace(min_stddev, max_stddev, num_stddevs)
    scales = np.linspace(min_scale, max_scale, num_scales)
    relative_errors= np.empty((num_stddevs, num_scales))

    #mat = np.asarray(Image.open('res/images/raccoon.jpg').convert('L')).astype(np.float64)
    #mat = mat[:size[0], :size[1]]
    mat = np.random.normal(scale=128, loc=128, size=size)
    watermark = np.random.normal(scale=128, loc=128, size=size)


    # Perturb watermarked matrix
    for i in range(num_stddevs):
        for j in range(num_scales):
            std_dev = std_devs[i]
            scale = scales[j]

            # Embed watermark
            mat_watermarked, vh= embed_watermark_jain_mod(mat, watermark, scale)
            
            # Perturb watermarked matrix
            perturbation = np.random.normal(scale=std_dev, size=size)
            mat_watermarked+= perturbation

            # Extract watermark and compute relative error
            watermark_extracted= extract_watermark_jain_mod(mat_watermarked, 
                    mat, vh, scale)
            relative_errors[i,j] = la.norm(watermark - watermark_extracted) / la.norm(watermark)

    fig, ax = plt.subplots()
    for i in range(num_scales):
        ax.plot(std_devs, relative_errors[:, i], label=r'a=%.2f' % scales[i])
    ax.legend()
    ax.set_title('Relative Errors in Watermark Extraction from Perturbed Image (Jain)')
    ax.set_xlabel('Standard Deviation of Perturbation Noise')
    ax.set_ylabel('Relative Error')

    # Save figure
    fig.set_size_inches(5,5)
    #fig.savefig('out/watermark/watermark_random_perturb_errors.png', bbox_inches='tight', layout='landscape')

    fig.show()

def plot_watermark_perturb_errors_scales_random_in_img_jain(min_scale=.01, max_scale=1, num_scales=100, min_stddev=.1, max_stddev=2, num_stddevs=20, 
        size=(128,128)):
    std_devs = np.linspace(min_stddev, max_stddev, num_stddevs)
    scales = np.linspace(min_scale, max_scale, num_scales)
    relative_errors= np.empty((num_stddevs, num_scales))

    #mat = np.asarray(Image.open('res/images/raccoon.jpg').convert('L')).astype(np.float64)
    #mat = mat[:size[0], :size[1]]
    mat = np.random.normal(scale=128, loc=128, size=size)
    watermark = np.random.normal(scale=128, loc=128, size=size)


    # Perturb watermarked matrix
    for i in range(num_stddevs):
        for j in range(num_scales):
            std_dev = std_devs[i]
            scale = scales[j]

            # Embed watermark
            mat_watermarked, vh= embed_watermark_jain(mat, watermark, scale)
            
            # Perturb watermarked matrix
            perturbation = np.random.normal(scale=std_dev, size=size)
            mat_watermarked+= perturbation

            # Extract watermark and compute relative error
            watermark_extracted= extract_watermark_jain(mat_watermarked, 
                    mat, vh, scale)
            relative_errors[i,j] = la.norm(watermark - watermark_extracted) / la.norm(watermark)

    fig, ax = plt.subplots()
    for i in range(num_scales):
        ax.plot(std_devs, relative_errors[:, i], label=r'a=%.2f' % scales[i])
    ax.legend()
    ax.set_title('Relative Errors in Watermark Extraction from Perturbed Image (Jain)')
    ax.set_xlabel('Standard Deviation of Perturbation Noise')
    ax.set_ylabel('Relative Error')

    # Save figure
    fig.set_size_inches(5,5)
    #fig.savefig('out/watermark/watermark_random_perturb_errors.png', bbox_inches='tight', layout='landscape')

    fig.show()



def plot_watermark_perceptibility_scales_jains(min_scale=.1, max_scale=1, num_scales=10, size=(128,128)):
    scales = np.linspace(min_scale, max_scale, num_scales)
    relative_errors_jain = np.empty(num_scales)
    relative_errors_jain_mod = np.empty(num_scales)

    #mat = np.asarray(Image.open('res/images/raccoon.jpg').convert('L')).astype(np.float64)
    #mat = mat[:size[0], :size[1]]
    mat = np.random.normal(scale=128, loc=128, size=size)
    watermark = np.random.normal(scale=128, loc=128, size=size)
    perturbation = np.random.normal(scale=10, size=size)

    # Perturb watermarked matrix
    for i in range(num_scales):
        scale = scales[i]

        # Embed watermark
        mat_watermarked_jain, vh_jain = embed_watermark_jain(mat, watermark, scale)
        mat_watermarked_jain_mod, vh_jain_mod = embed_watermark_jain_mod(mat, watermark, scale)

        # Compute relative error in watermarked matrix
        #relative_errors_jain[i] = la.norm(mat - mat_watermarked_jain) / la.norm(mat)
        #relative_errors_jain_mod[i] = la.norm(mat - mat_watermarked_jain_mod) / la.norm(mat)
        relative_errors_jain[i] = imgt.psnr(mat, mat_watermarked_jain)
        relative_errors_jain_mod[i] = imgt.psnr(mat, mat_watermarked_jain_mod)

    fig, ax = plt.subplots()
    ax.plot(scales, relative_errors_jain, label='Jain')
    ax.plot(scales, relative_errors_jain_mod, label='Jain Mod')

    ax.legend()
    ax.set_title('Relative Errors in Watermarked Image vs Original Image')
    ax.set_xlabel('Watermark Embedding Scaling Factor')
    ax.set_ylabel('Relative Error')

    # Save figure
    fig.set_size_inches(5,5)
    #fig.savefig('out/watermark/watermark_random_perturb_errors.png', bbox_inches='tight', layout='landscape')

    plt.show()

