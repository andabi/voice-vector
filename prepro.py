# -*- coding: utf-8 -*-
#!/usr/bin/env python

from pydub import AudioSegment
import os
import librosa
import soundfile as sf
import numpy as np


def read_wav(path, sr, duration=None, mono=True):
    wav, _ = librosa.load(path, mono=mono, sr=sr, duration=duration)
    return wav


def write_wav(wav, sr, path, format='wav', subtype='PCM_16'):
    sf.write(path, wav, sr, format=format, subtype=subtype)


def read_mfcc(prefix):
    filename = '{}.mfcc.npy'.format(prefix)
    mfcc = np.load(filename)
    return mfcc


def write_mfcc(prefix, mfcc):
    filename = '{}.mfcc'.format(prefix)
    np.save(filename, mfcc)


def read_spectrogram(prefix):
    filename = '{}.spec.npy'.format(prefix)
    spec = np.load(filename)
    return spec


def write_spectrogram(prefix, spec):
    filename = '{}.spec'.format(prefix)
    np.save(filename, spec)


def split_wav(wav, top_db):
    intervals = librosa.effects.split(wav, top_db=top_db)
    wavs = map(lambda i: wav[i[0]: i[1]], intervals)
    return wavs


def crop_random_wav(wav, length):
    """
    Randomly cropped a part in a wav file.
    :param wav: a waveform
    :param length: length to be randomly cropped.
    :return: a randomly cropped part of wav.
    """
    assert (wav.ndim <= 2)
    assert (type(length) == int)

    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - length)), 1)[0]
    end = start + length
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def rewrite_mp3_to_wav(src_path, tar_path):
    """
    Read mp3 file from source path, convert it to wav and write it to target path. 
    Necessary libraries: ffmpeg, libav.

    :param src_path: source mp3 file path
    :param tar_path: target wav file path
    """
    basepath, filename = os.path.split(src_path)
    os.chdir(basepath)
    AudioSegment.from_mp3(src_path).export(tar_path, format='wav')


def rewrite_decibel(src_path, tar_path, target_dB):
    """
    Read a wav, change its average amplitude to target decibel and write it to target path.
    :param src_path: source wav file path
    :param tar_path: target wav file path
    :param target_dB: target decibel
    """
    sound = AudioSegment.from_wav(src_path)
    change_dBFS = target_dB - sound.dBFS
    normalized_sound = sound.apply_gain(change_dBFS)
    basepath, filename, _ = _split_path(src_path)
    normalized_sound.export('{}/{}.wav'.format(tar_path, filename), 'wav')


def _split_path(path):
    """
    Split path to basename, filename and extension. For example, 'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: file path
    :return: basename, filename, and extension
    """
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension