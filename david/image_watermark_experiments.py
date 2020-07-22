import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import svd_tools as svdt
import watermark as wm

from PIL import Image

import png

from image_tools import *



def test_watermark(mode='randomized', rank=10, scale=1):
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))
    img_watermarked, watermarked_u, mat_s, watermarked_vh = embed_watermark(img, watermark, 
            scale=scale)
    watermark_extracted = extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh,
            scale=scale, mode=mode, rank=rank)
    plt.imshow(watermark_extracted)
    plt.show()

def img_watermark_plot():
    fig, axs = plt.subplots(1, 2)

    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))

    axs[0].imshow(img); axs[0].axis('off'); axs[0].set_title('Image')
    axs[1].imshow(watermark); axs[1].axis('off'); axs[1].set_title('Watermark')

    fig.set_size_inches(4, 2)
    fig.tight_layout()
    fig.savefig('out/watermark/img_watermark.png', transparent=True, bbox_inches='tight')

def test_watermark_scales_original(img, watermark, scales, size):
    fig, axs = plt.subplots(*size)

    original_ax = axs.flat[0]
    original_ax.imshow(img)
    original_ax.axis('off')
    original_ax.set_title('Original')

    for (ax, scale) in zip(axs.flat[1:], scales):
        ax.imshow(embed_watermark(img, watermark, scale)[0])
        ax.axis('off')
        ax.set_title('Watermarked (a=%.2f)' % scale)

    plt.show()

def test_watermark_scales(img, watermark, scales, size):
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))
    fig, axs = plt.subplots(*size)

    for (ax, scale) in zip(axs.flat, scales):
        ax.imshow(embed_watermark(img, watermark, scale)[0])
        ax.axis('off')
        ax.set_title('Watermarked (a=%.2f)' % scale)

    plt.show()

def test_watermark_extract_randomized(scales, ranks, mode='randomized'):
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))
    size = (len(scales), len(ranks))
    fig, axs = plt.subplots(*size)

    for i in range(len(scales)):
        img_watermarked, u, s, vh = embed_watermark(img, watermark, scales[i])
        for j in range(len(ranks)):
            axs[i, j].imshow(extract_watermark(img_watermarked, u, s, vh, scale=scales[i],
                mode=mode, rank=ranks[j], size=watermark.shape))
            axs[i,j].axis('off')
            axs[i,j].set_title('a=%.2f, rank=%d' % (scales[i], ranks[j]))

    fig.set_size_inches(7, 6)
    fig.tight_layout()
    fig.savefig('out/watermark/watermark_extraction_%s.png' % mode, transparent=True, bbox_inches='tight')
    
    plt.show()


def p_test_watermark_scales(scales, size):
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))
    fig, axs = plt.subplots(*size)

    for (ax, scale) in zip(axs.flat, scales):
        ax.imshow(embed_watermark(img, watermark, scale)[0])
        ax.axis('off')
        ax.set_title('a=%.2f' % scale)

    fig.set_size_inches(15/3., 10/3.)
    #fig.tight_layout()
    fig.savefig('out/watermark/watermark_scales.png', transparent=True, bbox_inches='tight')
    #plt.show()

def test_watermark_rotate(angle=45):
    scale = 0.1
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))
    img_watermarked, u, s, vh = embed_watermark(img, watermark, scale=scale)
    img_watermarked_obj = Image.fromarray(img_watermarked.astype('uint8')).convert('RGB')
    img_watermarked_obj_rotated = img_watermarked_obj.rotate(angle=angle)
    img_watermarked_rotated = np.asarray(img_watermarked_obj_rotated)
    watermark_extracted = extract_watermark(img_watermarked_rotated, u, s, vh, scale=scale)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(img); axs[0, 0].axis('off'); axs[0,0].set_title('Original')
    axs[0, 1].imshow(watermark); axs[0, 1].axis('off'); axs[0,1].set_title('Watermark')
    axs[1, 0].imshow(img_watermarked_rotated); axs[1, 0].axis('off'); axs[1,0].set_title('Watermarked Image, Rotated')
    axs[1, 1].imshow(watermark_extracted); axs[1, 1].axis('off'); axs[1,1].set_title('Extracted Watermark')


    #fig.set_size_inches(15/3., 10/3.)
    #fig.tight_layout()
    #fig.savefig('out/watermark/watermark_scales.png', transparent=True, bbox_inches='tight')
    plt.show()

def show_watermark_jain_scales():
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))

    img_watermarked_0, watermark_vh_1 = embed_watermark_jain(img, watermark, 0.25)
    img_watermarked_1, watermark_vh_1 = embed_watermark_jain(img, watermark, 0.1)
    img_watermarked_2, watermark_vh_2 = embed_watermark_jain(img, watermark, 0.01)

    img_watermarked_0 = np.clip(np.floor(img_watermarked_0), 0, 255).astype(np.uint8)
    img_watermarked_1 = np.clip(np.floor(img_watermarked_1), 0, 255).astype(np.uint8)
    img_watermarked_2 = np.clip(np.floor(img_watermarked_2), 0, 255).astype(np.uint8)

    Image.fromarray(img_watermarked_0).convert('RGB').save('out/watermark/raccoon_watermarked_jain_0.25.png')
    Image.fromarray(img_watermarked_1).convert('RGB').save('out/watermark/raccoon_watermarked_jain_0.1.png')
    Image.fromarray(img_watermarked_2).convert('RGB').save('out/watermark/raccoon_watermarked_jain_0.01.png')

def show_watermark_liutan_scales():
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))

    img_watermarked_0, _, _, _ = embed_watermark(img, watermark, 0.25)
    img_watermarked_1, _, _, _ = embed_watermark(img, watermark, 0.1)
    img_watermarked_2, _, _, _ = embed_watermark(img, watermark, 0.01)

    img_watermarked_0 = np.clip(np.floor(img_watermarked_0), 0, 255).astype(np.uint8)
    img_watermarked_1 = np.clip(np.floor(img_watermarked_1), 0, 255).astype(np.uint8)
    img_watermarked_2 = np.clip(np.floor(img_watermarked_2), 0, 255).astype(np.uint8)

    Image.fromarray(img_watermarked_0).convert('RGB').save('out/watermark/raccoon_watermarked_liutan_0.25.png')
    Image.fromarray(img_watermarked_1).convert('RGB').save('out/watermark/raccoon_watermarked_liutan_0.1.png')
    Image.fromarray(img_watermarked_2).convert('RGB').save('out/watermark/raccoon_watermarked_liutan_0.01.png')



def test_watermark_jain(scale=1):
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))

    img_watermarked, watermark_vh = embed_watermark_jain(img, watermark, scale)
    watermark_extracted = extract_watermark_jain(img_watermarked, img, watermark_vh, scale)

    img_watermarked = np.clip(np.floor(img_watermarked), 0, 255).astype(np.uint8)
    watermark_extracted = np.clip(watermark_extracted, 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(img); axs[0, 0].axis('off'); axs[0,0].set_title('Original')
    axs[0, 1].imshow(watermark); axs[0, 1].axis('off'); axs[0,1].set_title('Watermark')
    axs[1, 0].imshow(img_watermarked.astype(np.uint8)); axs[1, 0].axis('off'); axs[1,0].set_title('Watermarked Image')
    axs[1, 1].imshow(watermark_extracted.astype(np.uint8)); axs[1, 1].axis('off'); axs[1,1].set_title('Extracted Watermark')
    
    plt.show()

def test_wrong_watermark_jain(scale=0.01):
    raccoon = np.asarray(Image.open('res/images/raccoon.jpg'))
    redpanda = np.asarray(Image.open('res/images/redpanda.jpg'))
    fox = np.asarray(Image.open('res/images/fox.jpg'))

    raccoon_wm_redpanda, redpanda_vh = embed_watermark_jain(raccoon, redpanda, scale)
    raccoon_wm_fox, fox_vh = embed_watermark_jain(raccoon, fox, scale)
    fox_extracted_wrong = extract_watermark_jain(raccoon_wm_redpanda, raccoon, fox_vh, scale)

    raccoon_wm_redpanda = np.clip(np.floor(raccoon_wm_redpanda), 0, 255).astype(np.uint8)
    fox_extracted_wrong = np.clip(fox_extracted_wrong, 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].imshow(raccoon); axs[0, 0].axis('off'); axs[0,0].set_title('Original')
    axs[0, 1].imshow(redpanda); axs[0, 1].axis('off'); axs[0,1].set_title('Watermark')
    axs[0, 2].imshow(fox); axs[0, 2].axis('off'); axs[0,2].set_title('Wrong Watermark')
    axs[1, 0].imshow(raccoon_wm_redpanda); axs[1, 0].axis('off'); axs[1,0].set_title('Image Watermarked with Red Panda')
    axs[1,1].axis('off')
    axs[1, 2].imshow(fox_extracted_wrong); axs[1, 2].axis('off'); axs[1,2].set_title('Fox Watermark Extracted')

    fig.set_size_inches(10, 15)
    fig.tight_layout()
    
    plt.show()

def test_wrong_watermark_liutan(scale=0.01):
    raccoon = np.asarray(Image.open('res/images/raccoon.jpg'))
    redpanda = np.asarray(Image.open('res/images/redpanda.jpg'))
    fox = np.asarray(Image.open('res/images/fox.jpg'))

    raccoon_wm_redpanda, redpanda_u, redpanda_s, redpanda_vh = embed_watermark(raccoon, redpanda, scale)
    raccoon_wm_fox, fox_u, fox_s, fox_vh = embed_watermark(raccoon, fox, scale)
    fox_extracted_wrong = extract_watermark(raccoon_wm_redpanda, fox_u, fox_s, fox_vh, scale)

    raccoon_wm_redpanda = np.clip(np.floor(raccoon_wm_redpanda), 0, 255).astype(np.uint8)
    fox_extracted_wrong = np.clip(fox_extracted_wrong, 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].imshow(raccoon); axs[0, 0].axis('off'); axs[0,0].set_title('Original')
    axs[0, 1].imshow(redpanda); axs[0, 1].axis('off'); axs[0,1].set_title('Watermark')
    axs[0, 2].imshow(fox); axs[0, 2].axis('off'); axs[0,2].set_title('Wrong Watermark')
    axs[1, 0].imshow(raccoon_wm_redpanda); axs[1, 0].axis('off'); axs[1,0].set_title('Image Watermarked with Red Panda')
    axs[1,1].axis('off')
    axs[1, 2].imshow(fox_extracted_wrong); axs[1, 2].axis('off'); axs[1,2].set_title('Fox Watermark Extracted')

    fig.set_size_inches(10, 15)
    fig.tight_layout()
    
    plt.show()


def test_watermark_jain_gray(scale=1):
    img = np.asarray(Image.open('res/images/raccoon.jpg').convert('L'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg').convert('L'))

    img_watermarked, watermark_vh = wm.embed_watermark_jain(img, watermark, scale)
    watermark_extracted = wm.extract_watermark_jain(img_watermarked, img, watermark_vh, scale)

    img_watermarked = np.clip(np.floor(img_watermarked), 0, 255).astype(np.uint8)
    watermark_extracted = np.clip(watermark_extracted, 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(img, cmap='gray'); axs[0, 0].axis('off'); axs[0,0].set_title('Original')
    axs[0, 1].imshow(watermark, cmap='gray'); axs[0, 1].axis('off'); axs[0,1].set_title('Watermark')
    axs[1, 0].imshow(img_watermarked, cmap='gray'); axs[1, 0].axis('off'); axs[1,0].set_title('Watermarked Image')
    axs[1, 1].imshow(watermark_extracted, cmap='gray'); axs[1, 1].axis('off'); axs[1,1].set_title('Extracted Watermark')
    
    fig.show()

def test_watermark_jain_rotate(scale=0.1, angle=45):
    img = np.asarray(Image.open('res/images/raccoon.jpg').convert('L'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg').convert('L'))
    img_watermarked, vh = wm.embed_watermark_jain(img, watermark, scale=scale)
    print(img_watermarked.dtype)
    print(img_watermarked)
    img_watermarked_obj = Image.fromarray(img_watermarked.astype('float64'))#.convert('L')
    img_watermarked_obj_rotated = img_watermarked_obj#.rotate(angle=angle)
    img_watermarked_rotated = np.asarray(img_watermarked_obj_rotated)
    watermark_extracted = wm.extract_watermark_jain(img_watermarked_rotated, img, vh, scale=scale)
    print(watermark_extracted.dtype)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(img, cmap='gray'); axs[0, 0].axis('off'); axs[0,0].set_title('Original')
    axs[0, 1].imshow(watermark, cmap='gray'); axs[0, 1].axis('off'); axs[0,1].set_title('Watermark')
    axs[1, 0].imshow(img_watermarked_rotated, cmap='gray'); axs[1, 0].axis('off'); axs[1,0].set_title('Watermarked Image, Rotated')
    axs[1, 1].imshow(watermark_extracted, cmap='gray'); axs[1, 1].axis('off'); axs[1,1].set_title('Extracted Watermark')
    fig.show()







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

def plot_watermark_perturb_errors_sparse(scale=1, min_mult=1, max_mult=20, num_points=20, 
        size=(128,128)):
    mults = np.linspace(min_mult, max_mult, num_points)
    relative_errors_jain = np.empty_like(mults)
    relative_errors_liutan = np.empty_like(mults)

    #mat = np.asarray(Image.open('res/images/raccoon.jpg').convert('L')).astype(np.float64)
    #mat = mat[:size[0], :size[1]]
    mat = np.random.normal(scale=128, size=size)
    watermark = np.random.normal(scale=128, size=size)

    # Embed watermark
    mat_watermarked_jain, vh_jain = embed_watermark_jain(mat, watermark, scale)
    mat_watermarked_liutan, u_liutan, s_liutan, vh_liutan = embed_watermark(mat, watermark, scale)

    # Perturb watermarked matrix
    for i in range(num_points):
        #perturbation = np.random.normal(scale=std_devs[i], size=size)
        #perturbation -= np.max(perturbation)
        perturbation = sp.sparse.rand(*size, density=0.05).A * mults[i]

        watermark_extracted_jain = extract_watermark_jain(mat_watermarked_jain + perturbation, 
                mat, vh_jain, scale)
        relative_errors_jain[i] = la.norm(watermark - watermark_extracted_jain) / la.norm(watermark)

        watermark_extracted_liutan = extract_watermark(mat_watermarked_liutan + perturbation, 
                u_liutan, s_liutan, vh_liutan, scale)
        relative_errors_liutan[i] = la.norm(watermark - watermark_extracted_liutan) / la.norm(watermark)

    fig, ax = plt.subplots()
    ax.plot(mults, relative_errors_jain, label='Jain')
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
    relative_errors_liutan = np.empty_like(std_devs)

    #mat = np.asarray(Image.open('res/images/raccoon.jpg').convert('L')).astype(np.float64)
    #mat = mat[:size[0], :size[1]]
    mat = np.random.normal(scale=128, loc=128, size=size)
    watermark = np.random.normal(scale=128, loc=128, size=size)

    # Embed watermark
    mat_watermarked_jain, vh_jain = embed_watermark_jain(mat, watermark, scale)
    mat_watermarked_liutan, u_liutan, s_liutan, vh_liutan = embed_watermark(mat, watermark, scale)

    # Perturb watermarked matrix
    for i in range(num_points):
        perturbation = np.random.normal(scale=std_devs[i], size=size)

        watermark_extracted_jain = extract_watermark_jain(mat_watermarked_jain + perturbation, 
                mat, vh_jain, scale)
        relative_errors_jain[i] = la.norm(watermark - watermark_extracted_jain) / la.norm(watermark)

        watermark_extracted_liutan = extract_watermark(mat_watermarked_liutan + perturbation, 
                u_liutan, s_liutan, vh_liutan, scale)
        relative_errors_liutan[i] = la.norm(watermark - watermark_extracted_liutan) / la.norm(watermark)

    fig, ax = plt.subplots()
    ax.plot(std_devs, relative_errors_jain, label='Jain')
    ax.plot(std_devs, relative_errors_liutan, label='LiuTan')
    ax.legend()
    ax.set_title('Relative Errors in Watermark Extraction from Perturbed Image')
    ax.set_xlabel('Standard Deviation of Perturbation Noise')
    ax.set_ylabel('Relative Error')

    # Save figure
    fig.set_size_inches(5,5)
    fig.savefig('out/watermark/watermark_random_perturb_errors.png', bbox_inches='tight', layout='landscape')

    plt.show()



