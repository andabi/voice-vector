# -*- coding: utf-8 -*-
#!/usr/bin/env python

from pydub import AudioSegment
import os
import librosa
import soundfile as sf
import numpy as np
from scipy import signal


def read_wav(path, sr, duration=None, mono=True):
    wav, _ = librosa.load(path, mono=mono, sr=sr, duration=duration)
    return wav


def write_wav(wav, sr, path, format='wav', subtype='PCM_16'):
    sf.write(path, wav, sr, format=format, subtype=subtype)


def rewrite_mp3_to_wav(source_path, target_path):
    '''
    Necessary libraries: ffmpeg, libav
    :param source_path: 
    :param target_path: 
    :return: 
    '''
    basepath, filename = os.path.split(source_path)
    os.chdir(basepath)
    AudioSegment.from_mp3(source_path).export(target_path, format='wav')


def linear_to_mel(magnitude, sr, n_fft, n_mels):
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, magnitude)  # (n_mels, t) # mel spectrogram
    return mel


def amp_to_db(amp):
    return librosa.amplitude_to_db(amp)


def wav2spectrogram(wav, n_fft, win_length, hop_length):
    '''
    
    :param wav: 
    :param n_fft: 
    :param win_length: 
    :param hop_length: 
    :return: (1 + n_fft/2, t)
    '''
    spec = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return np.abs(spec), np.angle(spec)


def spectrogram2wav(mag, n_fft, win_length, hop_length, num_iters, phase_angle=None, length=None):
    '''
    
    :param mag: shape=(1 + n_fft/2, t)
    :param n_fft: 
    :param win_length: 
    :param hop_length: 
    :param num_iters: 
    :param phase_angle: 
    :param length: 
    :return: 
    '''
    assert (num_iters > 0)
    if phase_angle is None:
        phase_angle = np.pi * np.random.rand(*mag.shape)
    spec = mag * np.exp(1.j * phase_angle)
    for i in range(num_iters):
        wav = librosa.istft(spec, win_length=win_length, hop_length=hop_length, length=length)
        if i != num_iters - 1:
            spec = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            _, phase = librosa.magphase(spec)
            phase_angle = np.angle(phase)
            spec = mag * np.exp(1.j * phase_angle)
    return wav


def preemphasis(x, coeff=0.97):
    return signal.lfilter([1, -coeff], [1], x)


def inv_preemphasis(x, coeff=0.97):
    return signal.lfilter([1], [1, -coeff], x)


def split(wav, top_db):
    intervals = librosa.effects.split(wav, top_db=top_db)
    wavs = map(lambda i: wav[i[0]: i[1]], intervals)
    return wavs


def wav_random_crop(wav, length):
    assert(wav.ndim <= 2)
    assert(type(length) == int)

    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - length)), 1)[0]
    end = start + length
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav
