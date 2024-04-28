'''Based on: https://github.com/simonalexanderson/StyleGestures'''
import numpy as np
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os
import math
from tqdm import tqdm


def extract_melspec(audio_dir, files, destpath, fps, n_mels: int = 20):
    print('Extract MFCC... ...')
    for f in tqdm(files, desc=f'Extract MFCC with {fps}_fps_{n_mels}_mels'):
        file = os.path.join(audio_dir, f + '.wav')
        outfile = destpath + '/' + f + '.npy'

        # print('{}\t->\t{}'.format(file, outfile))
        # fs1, X1 = wav.read(file)
        X, fs = librosa.load(file, sr=None)
        # print("X1" + str(X1))
        # print("X" + str(X))
        # X1 = X1.astype(float) / math.pow(2, 15) # ????
        # print("X1 pow" + str(X1))
        assert fs % fps == 0

        hop_len = int(fs / fps)

        n_fft = int(fs * 0.13)
        C = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=2048, hop_length=hop_len, n_mels=n_mels, fmin=0.0,
                                           fmax=8000)
        C = np.log(C)
        np.save(outfile, np.transpose(C))
        print(f"\n audio feature {f}:=  {np.shape(np.transpose(C))}")


def extract_waveform(audiopath, files, destpath, fps, sample_rate):
    for f in tqdm(files, desc=f'Resampling all the Audios to {sample_rate}'):
        file = os.path.join(audiopath, f+'.wav')
        full_audio_data, sr = librosa.load(file, sr=sample_rate)
        outfile = os.path.join(destpath, f'{f}.npy')
        np.save(outfile, full_audio_data)


