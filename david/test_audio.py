import numpy as np
import numpy.linalg as la
import scipy.io.wavfile as wav
import scipy.signal as sig

import matplotlib.pyplot as plt
from svd_tools import *

audio_path = 'res/audio/'

def anti_approx_plot():
    # sample rate, sound
    sr_sound, sound = wav.read(audio_path + 'anti.wav')

    # STFT of sound
    Sound_freqs, Sound_times, Sound = sig.stft(sound)

    print('Rank %d' % la.matrix_rank(Sound))

    # Create figure
    fig, axs = plt.subplots(2,3)

    ranks = np.array([[100, 50, 30], [15, 5, 1]])

    # Approximations of sound STFT
    for row in range(0,2):
        for col in range(0, 3):
            t = (row, col)

            Sound_approx = rank_k_approx(Sound, rank=ranks[t])
            sound_approx_ts, sound_approx = sig.istft(Sound_approx)

            axs[t].set_title('Rank %d' % ranks[t])
            axs[t].specgram(sound_approx, Fs = sr_sound, NFFT=256, mode='magnitude')
            axs[t].set_xlabel('Time (s)')
            axs[t].set_ylabel('Frequency (Hz)')

            wav.write('out/audio/antidisestablishmentarianism/anti_approx_rank%d.wav' % ranks[t], sr_sound, sound_approx.astype(np.int16))
    fig.tight_layout()
    plt.savefig('out/audio/antidisestablishmentarianism/anti_approx_specgrams.png', bbox_inches='tight')
    plt.show()

def anti_approx_plot_randomized():
    # sample rate, sound
    sr_sound, sound = wav.read(audio_path + 'anti.wav')

    # STFT of sound
    Sound_freqs, Sound_times, Sound = sig.stft(sound)

    print('Rank %d' % la.matrix_rank(Sound))

    # Create figure
    fig, axs = plt.subplots(2,3)

    ranks = np.array([[100, 50, 30], [15, 5, 1]])

    # Approximations of sound STFT
    for row in range(0,2):
        for col in range(0, 3):
            t = (row, col)

            Sound_approx = rank_k_approx(Sound, rank=ranks[t], randomized=True, oversample=10)
            sound_approx_ts, sound_approx = sig.istft(Sound_approx)

            axs[t].set_title('Rank %d' % ranks[t])
            axs[t].specgram(sound_approx, Fs = sr_sound, NFFT=256, mode='magnitude')
            axs[t].set_xlabel('Time (s)')
            axs[t].set_ylabel('Frequency (Hz)')

            wav.write('out/audio/antidisestablishmentarianism/anti_approx_randomized_rank%d.wav' % ranks[t], sr_sound, sound_approx.astype(np.int16))
    fig.tight_layout()
    plt.savefig('out/audio/antidisestablishmentarianism/anti_approx_randomized_specgrams.png', bbox_inches='tight')
    plt.show()


#anti_approx_plot()
#anti_approx_plot_randomized()

sr_sound, sound = wav.read(audio_path + 'anti.wav')
print(np.mean(np.abs(sound)))



'''
def show_spectrogram(mat):
    if(mat.ndim == 1):
        f, t, Sound = sig.stft(mat)
    else:
        Sound = mat

    Sound_approx = rank_k_approx(Sound, min_energy=.6)
    spectrogram = np.absolute(Sound)
    spectrogram = np.absolute(Sound_approx)
    print(np.max(spectrogram))
    print(np.min(spectrogram))

    plt.imshow(spectrogram, cmap='plasma')
    plt.axis(False)
    plt.title('Original spectrogram')
    plt.show()

    plt.imshow(spectrogram**2, cmap='plasma')
    plt.axis(False)
    plt.title('Compressed Spectrogram')
    plt.show()
'''

'''
base_sr, base = wav.read('res/audio/librarian4.wav')
librarian_sr, librarian = wav.read('res/audio/librarian3.wav')

bf, bt, Base = sig.stft(base)
lf, lt, Librarian = sig.stft(librarian)

print(Base.shape)
print(Librarian.shape)

Librarian_approx = rank_k_approx(Librarian, rank=1)
lt_, librarian_approx = sig.istft(Librarian_approx)
wav.write('out/audio/librarian3_approx_rank1.wav', librarian_sr, librarian_approx.astype(np.int16))
'''

'''
Librarian_new = replace_singular_vectors(Base, Librarian, range(0,129), 'left')
Librarian_new_approx_rank10 = rank_k_approx(Librarian_new, rank=5)
lt_, librarian_new = sig.istft(Librarian_new)
lt_, librarian_new_approx_rank10 = sig.istft(Librarian_new_approx_rank10)
wav.write('out/audio/librarian4_librarian3_all_approx_rank5.wav', librarian_sr, librarian_new_approx_rank10.astype(np.int16))
'''



#Sound_approx = rank_k_approx(Sound, rank=200, randomized=True, oversample=10)


#ts, sound_new = sig.istft(Sound_new)

#wav.write('out/word_approx_randomized_energy0.9.wav', sample_rate, sound_new.astype(np.int16))
