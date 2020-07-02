#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:00:55 2020

@author: jenzyy
reference: https://stackoverflow.com/questions/53633177/how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numpy-array-to-mp3/53633178
"""


import pydub 
import numpy as np
import scipy.signal as sig
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")
  
def rand_svd(X,r,q,p):
    ny = X.shape[1]
    P =  np.random.standard_normal(size=(ny, r+p))
    Z = X @ P
    for k in range(1,q):
        Z = X @ (X.conj().transpose() @ Z)
    Q,R = np.linalg.qr(Z,mode='reduced')
    Y = Q.conj().transpose() @ X
    Uy , S , V = np.linalg.svd(Y, full_matrices=False)
    U = Q @ Uy
    approx = U @ np.diag(S)[:, :r] @ V[:r, :]
    return approx

def specgraph(data):
    mp3_audio = AudioSegment.from_file(data, format="mp3")  # read mp3
    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    FS, data = wavfile.read(wname)  # read wav file
    if mp3_audio.channels==2:
        plt.specgram(data[:,0], Fs=FS, NFFT=128, noverlap=0)  # plot
    else:
        plt.specgram(data, Fs=FS, NFFT=128, noverlap=0)  # plot
    plt.show()

# load mp3
sr, x = read('news.mp3')

f,t,mat = sig.stft(x[:,0])
f,t,mat1 = sig.stft(x[:,0])

approx = np.empty_like(mat)

q = 1
p = 5
r = 5

approx = rand_svd(mat, r, q, p)
approx1 = rand_svd(mat1, r, q, p)

ts,new = sig.istft(approx)
ts,new1 = sig.istft(approx1)

out = np.array([new,new1]).conj().transpose()

# output
write('news_rank_5.mp3', sr, out)
