import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import svd_tools as svdt
import watermark as wm

from PIL import Image

'''
Compresses an image using a low rank approximation. Takes in either the rank of the approximation or the compression ratio.
'''
def compress_image(img, ratio=None, rank=None, min_energy=None, mode='deterministic', oversample=0, power_iterations=0):
    return process_image((lambda x: svdt.rank_k_approx(x, rank=rank, ratio=ratio,
        oversample=oversample, min_energy=min_energy, 
        mode=mode, power_iterations=power_iterations)), img)

def embed_watermark(img, watermark, scale=1):
    img_type = img.dtype
    img = img.astype(np.float64)
        
    # Stack color channels
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    watermark_rows, watermark_columns = watermark.shape[:2] 
    watermark_stacked = watermark.reshape(watermark_rows, -1)
    
    # Embed watermark in image
    img_watermarked_stacked, watermarked_u, mat_s, watermarked_vh = wm.embed_watermark(img_stacked, watermark_stacked, scale)
    img_watermarked = img_watermarked_stacked.reshape(img_rows, img_columns, -1)

    # Get rid of redundant dimensions
    if img_watermarked.shape[2] == 1:
        img_watermarked.shape = img_watermarked.shape[:2]

    # Handle overflow/underflow issues
    if np.issubdtype(img_type, np.integer):
        img_watermarked = np.clip(img_watermarked, 0, 255)
    else:
        img_watermarked = np.clip(img_watermarked, 0, 1)

    img_watermarked = img_watermarked.astype(img_type)
    return img_watermarked, watermarked_u, mat_s, watermarked_vh

def extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh, scale, 
        randomized=False, rank=1, size=None):
    img_type = img_watermarked.dtype
    img_watermarked = img_watermarked.astype(np.float64)
        
    # Stack color channels
    img_rows, img_columns = img_watermarked.shape[:2] 
    img_watermarked_stacked = img_watermarked.reshape(img_rows, -1)
    
    # Extract watermark
    if randomized:
        _, watermarked_s, _ = svdt.randomized_svd(img_watermarked_stacked, rank=rank)
    else:
        _, watermarked_s, _ = la.svd(img_watermarked_stacked)
    mat_watermarked_rows, mat_watermarked_cols = img_watermarked_stacked.shape
    num_sv = len(watermarked_s)
    watermarked_s_matrix = np.pad(np.diag(watermarked_s), 
            [(0, mat_watermarked_rows - num_sv), (0, mat_watermarked_cols - num_sv)])

    watermark_stacked = (watermarked_u @ watermarked_s_matrix @ watermarked_vh - mat_s) / scale
    watermark = watermark_stacked.reshape(img_rows, img_columns, -1)

    # Get rid of redundant dimensions
    if watermark.shape[2] == 1:
        watermark.shape = watermark.shape[:2]

    # Handle overflow/underflow issues
    if np.issubdtype(img_type, np.integer):
        watermark = np.clip(watermark, 0, 255)
    else:
        watermark = np.clip(watermark, 0, 1)

    watermark = watermark.astype(img_type)
    if size is None:
        return watermark
    else:
        return watermark[:size[0], :size[1]]


def process_image(fun, img):
    img_type = img.dtype
    img = img.astype(np.float64)
        
    # Stack color channels
    rows, columns = img.shape[:2] 
    img_stacked = img.reshape(rows, -1)
    img_rank = np.linalg.matrix_rank(img_stacked)
    
    # Compute rank <rank> approximation of image
    fun_img_stacked = fun(img_stacked)
    fun_img = fun_img_stacked.reshape(rows, columns, -1)

    # Get rid of redundant dimensions
    if fun_img.shape[2] == 1:
        fun_img.shape = fun_img.shape[:2]

    # Handle overflow/underflow issues
    if np.issubdtype(img_type, np.integer):
        fun_img = np.clip(fun_img, 0, 255)
    else:
        fun_img = np.clip(fun_img, 0, 1)

    fun_img = fun_img.astype(img_type)
    return fun_img

def process_images(fun, imgs):
    img_type = img.dtype
    img = img.astype(np.float64)
        
    # Stack color channels
    rows, columns = img.shape[:2] 
    img_stacked = img.reshape(rows, -1)
    img_rank = np.linalg.matrix_rank(img_stacked)
    
    # Compute rank <rank> approximation of image
    fun_img_stacked = fun(img_stacked)
    fun_img = fun_img_stacked.reshape(rows, columns, -1)

    # Get rid of redundant dimensions
    if fun_img.shape[2] == 1:
        fun_img.shape = fun_img.shape[:2]

    # Handle overflow/underflow issues
    if np.issubdtype(img_type, np.integer):
        fun_img = np.clip(fun_img, 0, 255)
    else:
        fun_img = np.clip(fun_img, 0, 1)

    fun_img = fun_img.astype(img_type)
    return fun_img



def test_watermark(randomized=False, rank=10, scale=1):
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))
    img_watermarked, watermarked_u, mat_s, watermarked_vh = embed_watermark(img, watermark, 
            scale=scale)
    watermark_extracted = extract_watermark(img_watermarked, watermarked_u, mat_s, watermarked_vh,
            scale=scale, randomized=randomized, rank=rank)
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

def test_watermark_extract_randomized(scales, ranks):
    img = np.asarray(Image.open('res/images/raccoon.jpg'))
    watermark = np.asarray(Image.open('res/images/redpanda.jpg'))
    size = (len(scales), len(ranks))
    fig, axs = plt.subplots(*size)

    for i in range(len(scales)):
        img_watermarked, u, s, vh = embed_watermark(img, watermark, scales[i])
        for j in range(len(ranks)):
            axs[i, j].imshow(extract_watermark(img_watermarked, u, s, vh, scale=scales[i],
                randomized=True, rank=ranks[j], size=watermark.shape))
            axs[i,j].axis('off')
            axs[i,j].set_title('a=%.2f, rank=%d' % (scales[i], ranks[j]))

    fig.set_size_inches(7, 6)
    fig.tight_layout()
    fig.savefig('out/watermark/watermark_extraction_randomized.png', transparent=True, bbox_inches='tight')
    
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

