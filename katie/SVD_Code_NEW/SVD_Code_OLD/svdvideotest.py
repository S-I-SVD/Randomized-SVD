#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:35:55 2020

@author: katie
"""
import numpy as np
import matplotlib.pyplot as plt
import skvideo as vid
import skvideo.io
import skvideo.utils

import svd as svd

#loading the video

video = vid.io.vread('/Users/katie/Documents/surveillance_small_1.mp4')
a = video.shape
print(a)

videocompress = svd.regcompressvideo(video, 1)
plt.imshow(videocompress[0])
plt.show()
vid.io.vwrite('/Users/katie/Documents/surveillance_small_1_svd.mp4', videocompress)