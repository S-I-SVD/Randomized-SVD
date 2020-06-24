from matplotlib.pyplot import imread
import imageio 
import matplotlib.pyplot as plt
import numpy as np

# import image
im = imageio.imread("sky.jpg")

# rgb
red = im[:,:,0]
green = im[:,:,1]
blue = im[:,:,2]

def reg_svd(im,k):
    U, S, V = np.linalg.svd(im, full_matrices=False)
    approx = U @ np.diag(S)[:, :k] @ V[:k, :]
    return approx

approx = np.empty_like(im)
for k in[1,10,100,849]:
    approx[:,:,0]=reg_svd(red, k)
    approx[:,:,1]=reg_svd(green, k)
    approx[:,:,2]=reg_svd(blue, k)
    plt.imshow(approx)
    plt.show()


#def rgb2gray(rgb):
#    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#X = rgb2gray(im)
#U, S, V = np.linalg.svd(X, full_matrices=False)
#for k in [1,10,100, 849]:
#    approx = U @ np.diag(S)[:, :k] @ V[:k, :]
#    plt.imshow(approx, cmap = "gray")
 #   plt.show()
