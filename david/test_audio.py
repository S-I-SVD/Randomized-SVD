import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig

import matplotlib.pyplot as plt
from svd_tools import *

def show_spectrogram(mat):
    if(sound.ndim == 1):
        f, t, Sound = sig.stft(mat)
    else:
        Sound = mat
    spectrogram = np.absolute(Sound)**2
    print(np.max(spectrogram))
    print(np.min(spectrogram))
    plt.imshow(spectrogram, cmap='plasma')
    plt.show()

sample_rate, sound = wav.read('res/word.wav')
show_spectrogram(sound)
f, t, Sound = sig.stft(sound)

print(Sound.shape)

Sound_approx = rank_k_approx(Sound, rank=200, randomized=True, oversample=10)


ts, sound_new = sig.istft(Sound_new)

wav.write('out/word_approx_randomized_energy0.9.wav', sample_rate, sound_new.astype(np.int16))
