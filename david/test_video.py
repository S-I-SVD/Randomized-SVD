import numpy as np

import matplotlib.pyplot as plt

import skvideo as vid
import skvideo.io
import skvideo.utils


# Load video
video = vid.utils.rgb2gray(vid.io.vread('res/school_smol.mp4'))
video_shape = video.shape
num_frames = video_shape[0]
print('video loaded %s' % (video.shape,))

video_flattened = video.reshape(-1, num_frames)
print('video flattened %s' % (video_flattened.shape,))
plt.imshow(video_flattened)
plt.axis('off')
plt.show()

