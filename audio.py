# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import numpy as np
from scipy import signal


def wav2spec(wav, n_fft, win_length, hop_length, time_first=True):
    """
    Get magnitude and phase spectrogram from waveforms.

    Parameters
    ----------
    wav : np.ndarray [shape=(n,)]
        The real-valued waveform.

    n_fft : int > 0 [scalar]
        FFT window size.

    win_length  : int <= n_fft [scalar]
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

    hop_length : int > 0 [scalar]
        Number audio of frames between STFT columns.
        
    time_first : boolean. optional.
        if True, time axis is followed by bin axis. In this case, shape of returns is (t, 1 + n_fft/2)

    Returns
    -------
    mag : np.ndarray [shape=(t, 1 + n_fft/2) or (1 + n_fft/2, t)]
        Magnitude spectrogram.
        
    phase : np.ndarray [shape=(t, 1 + n_fft/2) or (1 + n_fft/2, t)]
        Phase spectrogram.

    """
    stft = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(stft)
    phase = np.angle(stft)

    if time_first:
        mag = mag.T
        phase = phase.T

    return mag, phase


def spec2wav(mag, n_fft, win_length, hop_length, num_iters=30, phase=None):
    """
    Get a waveform from the magnitude spectrogram by Griffin-Lim Algorithm.

    Parameters
    ----------
    mag : np.ndarray [shape=(1 + n_fft/2, t)]
        Magnitude spectrogram.

    n_fft : int > 0 [scalar]
        FFT window size.

    win_length  : int <= n_fft [scalar]
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

    hop_length : int > 0 [scalar]
        Number audio of frames between STFT columns.

    num_iters: int > 0 [scalar]
        Number of iterations of Griffin-Lim Algorithm.
        
    phase : np.ndarray [shape=(1 + n_fft/2, t)]
        Initial phase spectrogram.

    Returns
    -------
    wav : np.ndarray [shape=(n,)]
        The real-valued waveform.

    """
    assert (num_iters > 0)
    if phase is None:
        phase = np.pi * np.random.rand(*mag.shape)
    stft = mag * np.exp(1.j * phase)
    wav = None
    for i in range(num_iters):
        wav = librosa.istft(stft, win_length=win_length, hop_length=hop_length)
        if i != num_iters - 1:
            stft = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            _, phase = librosa.magphase(stft)
            phase = np.angle(phase)
            stft = mag * np.exp(1.j * phase)
    return wav


def preemphasis(wav, coeff=0.97):
    """
    Emphasize high frequency range of the waveform by increasing power(squared amplitude).
    
    Parameters
    ----------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.

    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.

    Returns
    -------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.
    """
    preem_wav = signal.lfilter([1, -coeff], [1], wav)
    return preem_wav


def inv_preemphasis(preem_wav, coeff=0.97):
    """
    Invert the pre-emphasized waveform to the original waveform.
    
    Parameters
    ----------
    preem_wav : np.ndarray [shape=(n,)]
        The pre-emphasized waveform.

    coeff: float <= 1 [scalar]
        Coefficient of pre-emphasis.

    Returns
    -------
    wav : np.ndarray [shape=(n,)]
        Real-valued the waveform.
    """
    wav = signal.lfilter([1], [1, -coeff], preem_wav)
    return wav


def linear_to_mel(linear, sr, n_fft, n_mels, **kwargs):
    """
    Convert a linear-spectrogram to mel-spectrogram.
    :param linear: Linear-spectrogram.
    :param sr: Sample rate.
    :param n_fft: FFT window size.
    :param n_mels: Number of mel filters.
    :return: Mel-spectrogram.
    """
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels, **kwargs)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, linear)  # (n_mels, t) # mel spectrogram
    return mel


def amp2db(amp):
    return librosa.amplitude_to_db(amp)


def db2amp(db):
    return librosa.db_to_amplitude(db)


def normalize_db(db, max_db, min_db):
    """
    Normalize dB-scaled spectrogram values to be in range of 0~1.
    :param db: Decibel-scaled spectrogram.
    :param max_db: Maximum dB.
    :param min_db: Minimum dB.
    :return: Normalized spectrogram.
    """
    norm_db = np.clip((db - min_db) / (max_db - min_db), 0, 1)
    return norm_db


def denormalize_db(norm_db, max_db, min_db):
    """
    Denormalize the normalized values to be original dB-scaled value.
    :param norm_db: Normalized spectrogram.
    :param max_db: Maximum dB.
    :param min_db: Minimum dB.
    :return: Decibel-scaled spectrogram.
    """
    db = np.clip(norm_db, 0, 1) * (max_db - min_db) + min_db
    return db


def dynamic_range_compression(db, threshold, ratio, method='downward'):
    """
    Execute dynamic range compression(https://en.wikipedia.org/wiki/Dynamic_range_compression) to dB.
    :param db: Decibel-scaled magnitudes
    :param threshold: Threshold dB
    :param ratio: Compression ratio.
    :param method: Downward or upward.
    :return: Range compressed dB-scaled magnitudes
    """
    if method is 'downward':
        db[db > threshold] = (db[db > threshold] - threshold) / ratio + threshold
    elif method is 'upward':
        db[db < threshold] = threshold - ((threshold - db[db < threshold]) / ratio)
    return db


def emphasize_magnitude(mag, power=1.2):
    """
    Emphasize a magnitude spectrogram by applying power function. This is used for removing noise.
    :param mag: magnitude spectrogram.
    :param power: exponent.
    :return: emphasized magnitude spectrogram.
    """
    emphasized_mag = np.power(mag, power)
    return emphasized_mag