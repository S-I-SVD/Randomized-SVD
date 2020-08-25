import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.sparse
import scipy.ndimage
import matplotlib.pyplot as plt

import svd_tools as svdt
import watermark as wm

from PIL import Image

import png

from image_tools import *
    
raccoon = load_image('res/public/raccoon.jpg')
fox = load_image('res/public/fox.jpg')
husky = load_image('res/public/husky.jpg')
noise = load_image('out/images/noise.jpg')
checker = load_image('res/images/checker.jpg')
checker_noise = load_image('res/images/checker_noise.jpg')

raccoon_f = raccoon.astype(np.float64)
fox_f = fox.astype(np.float64)
husky_f = husky.astype(np.float64)
noise_f = noise.astype(np.float64)


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
    watermark = np.asarray(Image.open('res/images/fox2.jpg'))
    img_watermarked, u, s, vh = embed_watermark(img, watermark, scale=scale)
    print(img_watermarked.shape)
    img_watermarked_rotated = scipy.ndimage.rotate(img_watermarked, angle, reshape=False)
    watermark_extracted = extract_watermark(img_watermarked_rotated, u, s, vh, scale=scale)

    img_watermarked_rotated = img_watermarked_rotated.clip(0, 255).astype('uint8')
    watermark_extracted = watermark_extracted.clip(0, 255).astype('uint8')

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(img); axs[0, 0].axis('off'); axs[0,0].set_title('Original')
    axs[0, 1].imshow(watermark); axs[0, 1].axis('off'); axs[0,1].set_title('Watermark')
    axs[1, 0].imshow(img_watermarked_rotated); axs[1, 0].axis('off'); axs[1,0].set_title('Watermarked Image, Rotated')
    axs[1, 1].imshow(watermark_extracted); axs[1, 1].axis('off'); axs[1,1].set_title('Extracted Watermark')

    #fig.set_size_inches(15/3., 10/3.)
    #fig.tight_layout()
    #fig.savefig('out/watermark/watermark_scales.png', transparent=True, bbox_inches='tight')
    plt.show()

def output_watermark_rotate(angles):
    scale = 0.1
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/fox2.jpg'))
    img_watermarked, u, s, vh = embed_watermark(img, watermark, scale=scale)

    for angle in angles:
        img_watermarked_rotated = scipy.ndimage.rotate(img_watermarked, angle, reshape=False)
        watermark_extracted = extract_watermark(img_watermarked_rotated, u, s, vh, scale=scale)

        img_watermarked_rotated = img_watermarked_rotated.clip(0, 255).astype('uint8')
        watermark_extracted = watermark_extracted.clip(0, 255).astype('uint8')

        Image.fromarray(img_watermarked_rotated).convert('RGB')\
            .save('out/images/raccoon_watermarked_fox_liutan_rotated%d.png' % angle)
        Image.fromarray(watermark_extracted).convert('RGB')\
            .save('out/images/raccoon_watermarked_fox_liutan_rotated%d_extracted.png' % angle)




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

def test_watermark_jain_perturb(scale=0.01):
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))

    img_watermarked, watermark_vh = embed_watermark_jain(img, watermark, scale)
    perturbation = np.random.normal(scale=1, size=img_watermarked.shape)
    watermark_extracted = extract_watermark_jain(img_watermarked + perturbation, img, watermark_vh, scale)

    img_watermarked = np.clip(np.floor(img_watermarked + perturbation), 0, 255).astype(np.uint8)
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

def test_wrong_watermark_jain_mod(scale=0.01):
    raccoon = np.asarray(Image.open('res/images/raccoon.jpg'))
    redpanda = np.asarray(Image.open('res/images/redpanda.jpg'))
    fox = np.asarray(Image.open('res/images/fox.jpg'))

    raccoon_wm_redpanda, redpanda_vh = embed_watermark_jain_mod(raccoon, redpanda, scale)
    raccoon_wm_fox, fox_vh = embed_watermark_jain_mod(raccoon, fox, scale)
    fox_extracted_wrong = extract_watermark_jain_mod(raccoon_wm_redpanda, raccoon, fox_vh, scale)

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
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/fox2.jpg'))

    # Embed watermark
    img_watermarked, vh = embed_watermark_jain(img, watermark, scale=scale)

    # Rotate watermarked image
    img_watermarked_obj = Image.fromarray(img_watermarked.clip(0,255).astype('uint8')).convert('RGB')
    img_watermarked_obj_rotated = img_watermarked_obj.rotate(angle=angle)
    img_watermarked_rotated = np.asarray(img_watermarked_obj_rotated)

    # Extract watermark
    watermark_extracted = extract_watermark_jain(img_watermarked_rotated, img, vh, scale=scale)\
        .clip(0,255).astype(np.uint8)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(img, cmap='gray'); axs[0, 0].axis('off'); axs[0,0].set_title('Original')
    axs[0, 1].imshow(watermark, cmap='gray'); axs[0, 1].axis('off'); axs[0,1].set_title('Watermark')
    axs[1, 0].imshow(img_watermarked_rotated, cmap='gray'); axs[1, 0].axis('off'); axs[1,0].set_title('Watermarked Image, Rotated')
    axs[1, 1].imshow(watermark_extracted, cmap='gray'); axs[1, 1].axis('off'); axs[1,1].set_title('Extracted Watermark')
    fig.show()

def test_watermark_jain_mod_rotate(scale=0.1, angle=45):
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/fox2.jpg'))

    # Embed watermark
    img_watermarked, vh = embed_watermark_jain_mod(img, watermark, scale=scale)

    # Rotate watermarked image
    img_watermarked_obj = Image.fromarray(img_watermarked.clip(0,255).astype('uint8')).convert('RGB')
    img_watermarked_obj_rotated = img_watermarked_obj.rotate(angle=angle)
    img_watermarked_rotated = np.asarray(img_watermarked_obj_rotated)

    # Extract watermark
    watermark_extracted = extract_watermark_jain_mod(img_watermarked_rotated, img, vh, scale=scale)\
        .clip(0,255).astype(np.uint8)

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

def visualize_jain_term(scale=0.1, mod=False, jain_term_title=None, img_watermarked_title=None,
        img1='raccoon.jpg', img2='fox2.jpg'):
    fig, ax = plt.subplots(2,2)

    img = np.asarray(Image.open('res/images/%s' % img1))
    if img2 == 'noise':
        watermark = np.random.randint(0, 256, size=img.shape, dtype=np.uint8)
    else:
        watermark = np.asarray(Image.open('res/images/%s' % img2))
    if mod:
        img_watermarked = embed_watermark_jain_mod(img, watermark, scale=scale)[0]
    else:
        img_watermarked = embed_watermark_jain(img, watermark, scale=scale)[0]
    jain_term = get_jain_term(img, watermark, mod=mod)

    #img_flattened = img.reshape(*(img.shape[:2]), -1)
    #jain_term_flattened = img.reshape(*(jain_term.shape[:2]), -1)
    #print(img_flattened.dtype)
    #print(jain_term.dtype)
    #print('relative error is %3f' % la.norm(img_flattened - jain_term_flattened)/la.norm(img_flattened))

    img_watermarked = img_watermarked.clip(0,255).astype(np.uint8)
    jain_term = jain_term.clip(0,255).astype(np.uint8)
    img = img.clip(0,255).astype(np.uint8)

    if jain_term_title != None:
        Image.fromarray(jain_term).convert('RGB').save('out/watermark/%s.png' % jain_term_title)
    if img_watermarked_title != None:
        Image.fromarray(img_watermarked).convert('RGB').save('out/watermark/%s.png' % img_watermarked_title)
    
    ax[0,0].imshow(img); ax[0,0].axis('off'); ax[0,0].set_title('Image')
    ax[0,1].imshow(watermark); ax[0,1].axis('off'); ax[0,1].set_title('Watermark')
    ax[1,0].imshow(img_watermarked); ax[1,0].axis('off'); ax[1,0].set_title('Watermarked Image (a=%.2f)' % scale)
    ax[1,1].imshow(jain_term); ax[1,1].axis('off'); ax[1,1].set_title('Term in Watermarking Scheme')

    plt.show()

def watermark_perceptibility_jains_matrix(scale=0.1):
    img_names = ['Raccoon', 'Fox', 'Husky', 'Noise']
    imgs = list(map(lambda name: np.asarray(Image.open('res/images/%s.jpg' % name)).astype(np.float64), ['raccoon', 'fox2', 'husky']))
    imgs.append(np.random.randint(0, 256, imgs[0].shape).astype(np.float64))

    watermark = np.asarray(Image.open('res/images/fox2.jpg')).astype(np.float64)

    num_imgs = len(imgs)
    print(num_imgs)
    output = np.empty((num_imgs, num_imgs, 2))

    for i in range(num_imgs):
        for j in range(num_imgs):
            img = imgs[i]
            watermark = imgs[j]

            # Embed watermark
            mat_watermarked_jain, vh_jain, jain_term = embed_watermark_jain(img, watermark, scale, 
                    term=True)
            mat_watermarked_jain_mod, vh_jain_mod, jain_mod_term = embed_watermark_jain_mod(img, 
                    watermark, scale, term=True)

            # Compute relative error in watermarked matrix
            img_stacked = img.reshape(*img.shape + (-1,))
            jain_term_stacked = jain_term.reshape(*jain_term.shape + (-1,)) 
            jain_mod_term_stacked = jain_mod_term.reshape(*jain_mod_term.shape + (-1,)) 

            output[i, j, 0] = la.norm(img_stacked - jain_term_stacked) / la.norm(img_stacked)
            output[i, j, 1] = la.norm(img_stacked - jain_mod_term_stacked) / la.norm(img_stacked)

    fig, ax = plt.subplots()
    differences = output[:, :, 0] - output[:, :, 1]
    print(differences)
    im = ax.imshow(differences)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(img_names)))
    ax.set_yticks(np.arange(len(img_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(img_names)
    ax.set_yticklabels(img_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(img_names)):
        for j in range(len(img_names)):
            text = ax.text(j, i, '%.2f' % (differences[i,j] / output[i,j,0]), ha="center", 
                va="center", color="w")

    ax.set_xlabel('Watermark')
    ax.set_ylabel('Image')

    #ax.set_title("")
    fig.tight_layout()
    plt.show()

def plot_watermark_perceptibility_scales_jains(img=raccoon_f, watermark=fox_f, min_scale=.01, max_scale=1, num_scales=50, fname=None):
    scales = np.linspace(min_scale, max_scale, num_scales)
    correlations_jain = np.empty(num_scales)
    correlations_jain_mod = np.empty(num_scales)

    #mat = np.asarray(Image.open('res/images/raccoon.jpg').convert('L')).astype(np.float64)
    #mat = mat[:size[0], :size[1]]
    #img = np.asarray(Image.open('res/images/raccoon.jpg'), dtype=np.float64)
    #watermark = np.asarray(Image.open('res/images/husky.jpg'), dtype=np.float64)
    #watermark = np.random.randint(0, 256, size=img.shape).astype(np.float64)

    # Perturb watermarked matrix
    for i in range(num_scales):
        scale = scales[i]

        # Embed watermark
        img_watermarked_jain, vh_jain = embed_watermark_jain(img, watermark, scale)
        img_watermarked_jain_mod, vh_jain_mod = embed_watermark_jain_mod(img, watermark, scale)

        # Compute relative error in watermarked matrix
        #relative_errors_jain[i] = la.norm(mat - mat_watermarked_jain) / la.norm(mat)
        #relative_errors_jain_mod[i] = la.norm(mat - mat_watermarked_jain_mod) / la.norm(mat)
        correlations_jain[i] = dot_angle(img, img_watermarked_jain)
        correlations_jain_mod[i] = dot_angle(img, img_watermarked_jain_mod)

    fig, ax = plt.subplots()
    ax.plot(scales, correlations_jain, label='Jain')
    ax.plot(scales, correlations_jain_mod, label='Jain Mod')

    ax.legend()
    #ax.set_title('Correlation Between Watermarked Image and Original Image')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'corr$(A, A_W)$')

    # Save figure
    fig.set_size_inches(5,5)
    if fname != None:
        fig.savefig('out/paper/watermark/%s.png' % fname, bbox_inches='tight', layout='landscape')

    plt.show()


def paper_liutan_scale_imgs():
    for alpha in ['1', '0.5', '0.25', '0.1']:
        raccoon_watermarked, u, s, vh = embed_watermark(raccoon, fox, scale=float(alpha))
        save_image(raccoon_watermarked, 'out/paper/watermark/raccoon_watermarked_fox_liutan_%s.png' % alpha)

def paper_jain_mod_scale_imgs():
    for alpha in ['1', '0.5', '0.25', '0.1']:
        raccoon_watermarked, vh = embed_watermark_jain_mod(raccoon, fox, scale=float(alpha))
        save_image(raccoon_watermarked, 'out/paper/watermark/jain_mod/raccoon_fox/raccoon_watermarked_fox_jain_mod_%s.png' % alpha)


def paper_liutan_rotate_extract(angles):
    scale = 0.1
    img = raccoon
    watermark = fox
    img_watermarked, u, s, vh = embed_watermark(img, watermark, scale=scale)

    for angle in angles:
        img_watermarked_rotated = scipy.ndimage.rotate(img_watermarked, angle, reshape=False)
        watermark_extracted = extract_watermark(img_watermarked_rotated, u, s, vh, scale=scale)

        img_watermarked_rotated = img_watermarked_rotated.clip(0, 255).astype('uint8')
        watermark_extracted = watermark_extracted.clip(0, 255).astype('uint8')

        Image.fromarray(img_watermarked_rotated).convert('RGB')\
            .save('out/paper/watermark/raccoon_watermarked_fox_liutan_rotated%d.png' % angle)
        Image.fromarray(watermark_extracted).convert('RGB')\
            .save('out/paper/watermark/raccoon_watermarked_fox_liutan_rotated%d_extracted.png' % angle)

def paper_liutan_phony_watermark():
    raccoon_fox = embed_watermark(raccoon, fox, scale=.1)[0]
    raccoon_husky, u, s, vh = embed_watermark(raccoon, husky, scale=.1)
    phony_husky = extract_watermark(raccoon_fox, u, s, vh, .1)
    save_image(phony_husky, 'out/paper/watermark/phony_husky.png')

def paper_jain_phony_watermark():
    raccoon_fox = embed_watermark_jain(raccoon, fox, scale=.1)[0]
    raccoon_husky, vh = embed_watermark_jain(raccoon, husky, scale=.1)
    phony_husky = extract_watermark_jain(raccoon_fox, raccoon, vh, .1)
    save_image(phony_husky, 'out/paper/watermark/jain/phony_husky_jain.png')

def paper_jain_mod_phony_watermark():
    raccoon_fox = embed_watermark_jain_mod(raccoon, fox, scale=.1)[0]
    raccoon_husky, vh = embed_watermark_jain(raccoon, husky, scale=.1)
    phony_husky = extract_watermark_jain_mod(raccoon_fox, raccoon, vh, 0.1)
    save_image(phony_husky, 'out/paper/watermark/jain_mod/phony_husky_jain_mod.png')



def paper_jain_scale_imgs():
    for alpha in ['1', '0.5', '0.25', '0.1', '0.01']:
        raccoon_watermarked, vh = embed_watermark_jain(raccoon, fox, scale=float(alpha))
        save_image(raccoon_watermarked, 'out/paper/watermark/raccoon_watermarked_fox_jain_%s.png' % alpha)

def paper_liutan_rotate_extract(angles):
    scale = 0.1
    img = raccoon
    watermark = fox
    img_watermarked, u, s, vh = embed_watermark(img, watermark, scale=scale)

    for angle in angles:
        img_watermarked_rotated = scipy.ndimage.rotate(img_watermarked, angle, reshape=False)
        watermark_extracted = extract_watermark(img_watermarked_rotated, u, s, vh, scale=scale)

        img_watermarked_rotated = img_watermarked_rotated.clip(0, 255).astype('uint8')
        watermark_extracted = watermark_extracted.clip(0, 255).astype('uint8')

        Image.fromarray(img_watermarked_rotated).convert('RGB')\
            .save('out/paper/watermark/raccoon_watermarked_fox_liutan_rotated%d.png' % angle)
        Image.fromarray(watermark_extracted).convert('RGB')\
            .save('out/paper/watermark/raccoon_watermarked_fox_liutan_rotated%d_extracted.png' % angle)

def paper_jain_rotate_extract(angles):
    scale = 0.25
    img = raccoon
    watermark = fox
    img_watermarked, vh = embed_watermark_jain(img, watermark, scale=scale)

    for angle in angles:
        img_watermarked_rotated = scipy.ndimage.rotate(img_watermarked, angle, 
                reshape=False)
        watermark_extracted = extract_watermark_jain(img_watermarked_rotated, 
                img, vh, scale=scale)

        img_watermarked_rotated = img_watermarked_rotated.clip(0, 255).astype('uint8')
        watermark_extracted = watermark_extracted.clip(0, 255).astype('uint8')

        Image.fromarray(img_watermarked_rotated).convert('RGB')\
            .save('out/paper/watermark/jain/rotate/raccoon_watermarked_fox_jain_rotated%d.png' % angle)
        Image.fromarray(watermark_extracted).convert('RGB')\
            .save('out/paper/watermark/jain/rotate/raccoon_watermarked_fox_jain_rotated%d_extracted.png' % angle)

def paper_jain_mod_rotate_extract(angles):
    scale = 0.25
    img = raccoon
    watermark = fox
    img_watermarked, vh = embed_watermark_jain_mod(img, watermark, scale=scale)

    for angle in angles:
        img_watermarked_rotated = scipy.ndimage.rotate(img_watermarked, angle, 
                reshape=False)
        watermark_extracted = extract_watermark_jain_mod(img_watermarked_rotated, 
                img, vh, scale=scale)

        img_watermarked_rotated = img_watermarked_rotated.clip(0, 255).astype('uint8')
        watermark_extracted = watermark_extracted.clip(0, 255).astype('uint8')

        Image.fromarray(img_watermarked_rotated).convert('RGB')\
            .save('out/paper/watermark/jain_mod/rotate/raccoon_watermarked_fox_jain_mod_rotated%d.png' % angle)
        Image.fromarray(watermark_extracted).convert('RGB')\
            .save('out/paper/watermark/jain_mod/rotate/raccoon_watermarked_fox_jain_mod_rotated%d_extracted.png' % angle)

def paper_jain_term():
    jain_term = embed_watermark_jain(raccoon, fox, scale=0.1, term=True)[2]
    save_image(jain_term, 'out/paper/watermark/jain/jain_term.png')

def paper_jain_mod_term():
    jain_term = embed_watermark_jain_mod(raccoon, fox, scale=0.1, term=True)[2]
    save_image(jain_term, 'out/paper/watermark/jain_mod/jain_mod_term.png')


def paper_sv_cumsum_plots():
    svdt.sv_cumsum_plot(raccoon.reshape(raccoon.shape[0], -1), 'out/paper/raccoon_cumsum.png')
    svdt.sv_cumsum_plot(noise.reshape(noise.shape[0], -1), 'out/paper/noise_cumsum.png')

def paper_sv_plots():
    svdt.sv_plot(raccoon.reshape(raccoon.shape[0], -1), 'out/paper/raccoon_sv.png')
    svdt.sv_plot(noise.reshape(noise.shape[0], -1), 'out/paper/noise_sv.png')
    svdt.sv_plot(checker.reshape(checker.shape[0], -1), 'out/paper/checker_sv.png')
    svdt.sv_plot(checker_noise.reshape(checker_noise.shape[0], -1), 'out/paper/checker_noise_sv.png')
    
def dot_angle(mat1, mat2):
    return np.sum(mat1 * mat2) / (np.sqrt(np.sum(mat1**2)) * np.sqrt(np.sum(mat2**2)))
