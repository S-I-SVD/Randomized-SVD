import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import skvideo as vid
import skvideo.io
import skvideo.utils

import svd_tools as svd

# Load video
video = vid.io.vread('res/doggo.mp4')

# Compress video
print('Compressing randomized...')
video_compressed_randomized = svd.compress_video(video, rank=5, randomized=True, oversample=10)
print('Compressing...')
video_compressed = svd.compress_video(video, rank=5)

# Save compressed video
vid.io.vwrite('out/doggo_rank5_randomized_oversample10.mp4', video_compressed_randomized)
vid.io.vwrite('out/doggo_rank5.mp4', video_compressed)

#plt.imshow(video_compressed[0])
#plt.show()
