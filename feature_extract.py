# -*- coding: utf-8 -*-
#!/usr/bin/env python

import librosa
import numpy as np

from audio import wav2spec, linear_to_mel, preemphasis, normalize_db


def wav2melspec(wav, sr, n_fft, win_length, hop_length, n_mels, time_first=True, **kwargs):

    # Linear spectrogram
    mag_spec, phase_spec = wav2spec(wav, n_fft, win_length, hop_length, time_first=False)

    # Mel-spectrogram
    mel_spec = linear_to_mel(mag_spec, sr, n_fft, n_mels, **kwargs)

    # Time-axis first
    if time_first:
        mel_spec = mel_spec.T  # (t, n_mels)

    return mel_spec


def wav2melspec_db(wav, sr, n_fft, win_length, hop_length, n_mels, normalize=False, max_db=None, min_db=None, time_first=True, **kwargs):

    # Mel-spectrogram
    mel_spec = wav2melspec(wav, sr, n_fft, win_length, hop_length, n_mels, time_first=False, **kwargs)

    # Decibel
    mel_db = librosa.amplitude_to_db(mel_spec)

    # Normalization
    mel_db = normalize_db(mel_db, max_db, min_db) if normalize else mel_db

    # Time-axis first
    if time_first:
        mel_db = mel_db.T  # (t, n_mels)

    return mel_db


def wav2mfcc(wav, sr, n_fft, win_length, hop_length, n_mels, n_mfccs, preemphasis_coeff=0.97, time_first=True, **kwargs):

    # Pre-emphasis
    wav_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Decibel-scaled mel-spectrogram
    mel_db = wav2melspec_db(wav_preem, sr, n_fft, win_length, hop_length, n_mels, time_first=False, **kwargs)

    # MFCCs
    mfccs = np.dot(librosa.filters.dct(n_mfccs, mel_db.shape[0]), mel_db)

    # Time-axis first
    if time_first:
        mfccs = mfccs.T  # (t, n_mfccs)

    return mfccs