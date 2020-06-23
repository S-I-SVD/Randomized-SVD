import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import skvideo as vid
import skvideo.io
import skvideo.utils

import svd_tools as svd

# Load video
video = vid.io.vread('res/school2.mp4')

# Compress video
video_compressed = svd.compress_video(video, ratio=.2, randomized=True, oversample=10)

# Save compressed video
vid.io.vwrite('out/school2_compressed.mp4', video_compressed)

plt.imshow(video_compressed[0])
plt.show()
