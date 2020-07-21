import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import svd_tools as svdt

def embed_watermark(mat, watermark, scale=1):
    mat_rows, mat_columns = mat.shape
    watermark_rows, watermark_columns = watermark.shape

    if mat_rows < watermark_rows or mat_columns < watermark_columns:
        print('Watermark must be smaller than matrix')
        return

    mat_u, mat_s, mat_vh = la.svd(mat)
    mat_num_sv = len(mat_s)
    
    # Compute the rectangular "diagonal" singular value matrix
    mat_s_matrix = np.pad(np.diag(mat_s), 
            [(0, mat_rows - mat_num_sv), (0, mat_columns - mat_num_sv)])
    watermark_padded = np.pad(watermark, 
            [(0, mat_rows - watermark_rows), (0, mat_columns - watermark_columns)])
    mat_s_matrix_watermarked = mat_s_matrix + scale * watermark_padded

    watermarked_u, watermarked_s, watermarked_vh = la.svd(mat_s_matrix_watermarked)
    watermarked_num_sv = len(watermarked_s)
    watermarked_s_matrix = np.pad(np.diag(watermarked_s), 
            [(0, mat_rows - watermarked_num_sv), (0, mat_columns - watermarked_num_sv)])

    mat_watermarked = mat_u @ watermarked_s_matrix @ mat_vh

    return mat_watermarked, watermarked_u, mat_s_matrix, watermarked_vh

def extract_watermark(mat_watermarked, watermarked_u, mat_s_matrix, watermarked_vh, scale):
    _, watermarked_s, _ = la.svd(mat_watermarked)
    mat_watermarked_rows, mat_watermarked_cols = mat_watermarked.shape
    num_sv = len(watermarked_s)
    watermarked_s_matrix = np.pad(np.diag(watermarked_s), 
            [(0, mat_watermarked_rows - num_sv), (0, mat_watermarked_cols - num_sv)])
    return (watermarked_u @ watermarked_s_matrix @ watermarked_vh - mat_s_matrix) / scale

'''
def embed_watermark_img(img, watermark, scale=1):
    img_type = img.dtype
    img = img.astype(np.float64)
        
    # Stack color channels
    img_rows, img_columns = img.shape[:2] 
    img_stacked = img.reshape(img_rows, -1)
    watermark_rows, watermark_columns = watermark.shape[:2] 
    watermark_stacked = watermark.reshape(watermark_rows, -1)
    
    # Embed watermark in image
    img_watermarked_stacked, watermarked_u, mat_s, watermarked_vh = embed_watermark(img_stacked, watermark_stacked, scale)
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

def extract_watermark(mat_watermarked, watermarked_u, mat_s, watermarked_vh):
    mat_watermarked_u, mat_watermarked_s, mat_watermarked_vh = la.svd(mat_watermarked)


def test_watermarks_img(img, watermark, scales, size):
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
'''
