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


def trim_wav(wav):
    wav, _ = librosa.effects.trim(wav)
    return wav


def fix_length(wav, length):
    if len(wav) != length:
        wav = librosa.util.fix_length(wav, length)
    return wav


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


def mp3_to_wav(src_path, tar_path):
    """
    Read mp3 file from source path, convert it to wav and write it to target path. 
    Necessary libraries: ffmpeg, libav.

    :param src_path: source mp3 file path
    :param tar_path: target wav file path
    """
    basepath, filename = os.path.split(src_path)
    os.chdir(basepath)
    AudioSegment.from_mp3(src_path).export(tar_path, format='wav')


def prepro_audio(source_path, target_path, format=None, sr=None, db=None):
    """
    Read a wav, change sample rate, format, and average decibel and write to target path.
    :param source_path: source wav file path
    :param target_path: target wav file path
    :param sr: sample rate.
    :param format: output audio format.
    :param db: decibel.
    """
    sound = AudioSegment.from_file(source_path, format)
    if sr:
        sound = sound.set_frame_rate(sr)
    if db:
        change_dBFS = db - sound.dBFS
        sound = sound.apply_gain(change_dBFS)
    sound.export(target_path, 'wav')


def _split_path(path):
    """
    Split path to basename, filename and extension. For example, 'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: file path
    :return: basename, filename, and extension
    """
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension