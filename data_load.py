# -*- coding: utf-8 -*-
# !/usr/bin/env python


import glob
import os
import random

import tensorflow as tf
from hparam import Hparam

from audio import read_wav, wav_random_crop, wav2spectrogram, linear_to_mel, amp_to_db
import numpy as np
import sys


class DataLoader:

    def __init__(self, data_path, sr, duration, n_fft, n_mels, win_length, hop_length, batch_size):
        self.data_path = data_path
        self.sr = sr
        self.duration = duration
        self.batch_size = batch_size
        self.length = int(duration * sr)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.length_spec = self.length // hop_length + 1
        self.speaker_names = [speaker for speaker in os.listdir(self.data_path) if
                              os.path.isdir(os.path.join(self.data_path, speaker))]

    def load_triplet(self, speaker_name):
        speakers_except = list(set(self.speaker_names) - set(list([speaker_name])))
        the_other_speaker_name = random.choice(speakers_except)
        wav = self._get_random_wav(speaker_name)
        wav_pos = self._get_random_wav(speaker_name)
        wav_neg = self._get_random_wav(the_other_speaker_name)

        # wav to spectrogram
        spec, _ = wav2spectrogram(wav, self.n_fft, self.win_length, self.hop_length)  # (1 + n_fft/2, t)
        spec_pos, _ = wav2spectrogram(wav_pos, self.n_fft, self.win_length, self.hop_length)
        spec_neg, _ = wav2spectrogram(wav_neg, self.n_fft, self.win_length, self.hop_length)

        # Mel-spectrogram
        spec = linear_to_mel(spec, self.sr, self.n_fft, self.n_mels)  # (n_mel, t)
        spec_pos = linear_to_mel(spec_pos, self.sr, self.n_fft, self.n_mels)
        spec_neg = linear_to_mel(spec_neg, self.sr, self.n_fft, self.n_mels)

        # Decibel
        spec = amp_to_db(spec)
        spec_pos = amp_to_db(spec_pos)
        spec_neg = amp_to_db(spec_neg)

        return spec.T, spec_pos.T, spec_neg.T  # (t, n_mel)

    def _get_random_wav(self, speaker_name):
        wavfiles = glob.glob('{}/{}/*.wav'.format(self.data_path, speaker_name))
        wavfile = random.choice(wavfiles)
        randomly_cropped_wav = wav_random_crop(read_wav(wavfile, self.sr), length=int(self.sr * self.duration))
        return randomly_cropped_wav

    def get_batch_queue(self):
        hp = Hparam.get_global_hparam()

        speaker_names = tf.convert_to_tensor(self.speaker_names)
        speaker_name = tf.train.slice_input_producer([speaker_names], shuffle=True)[0]
        spec, spec_pos, spec_neg = tf.py_func(self.load_triplet, [speaker_name], (tf.float32, tf.float32, tf.float32))
        spec_batch, spec_pos_batch, spec_neg_batch, speaker_name_batch = tf.train.batch([spec, spec_pos, spec_neg, speaker_name],
                                                                    shapes=[
                                                                        (self.length_spec, self.n_mels),
                                                                        (self.length_spec, self.n_mels),
                                                                        (self.length_spec, self.n_mels), ()],
                                                                    num_threads=hp.data_load.num_threads,
                                                                    batch_size=self.batch_size,
                                                                    capacity=self.batch_size * hp.data_load.num_threads,
                                                                    dynamic_pad=True)
        return spec_batch, spec_pos_batch, spec_neg_batch, speaker_name_batch  # (n, t, n_mels)

    def get_batch_placeholder(self):
        spec_batch = tf.placeholder(tf.float32, shape=(self.batch_size, self.length_spec, self.n_mels), name='x')
        spec_pos_batch = tf.placeholder(tf.float32, shape=(self.batch_size, self.length_spec, self.n_mels), name='x_pos')
        spec_neg_batch = tf.placeholder(tf.float32, shape=(self.batch_size, self.length_spec, self.n_mels), name='x_neg')
        speaker_name_batch = tf.placeholder(tf.string, shape=(self.batch_size, ), name='speaker_name')
        return spec_batch, spec_pos_batch, spec_neg_batch, speaker_name_batch

    def get_batch(self):
        triplet_batch, speaker_name_batch = list(), list()
        for i in range(self.batch_size):
            speaker_name = random.choice(self.speaker_names)
            triplet = self.load_triplet(speaker_name)
            triplet_batch.append(triplet)
            speaker_name_batch.append(speaker_name)
        spec_batch, spec_pos_batch, spec_neg_batch = map(np.array, zip(*triplet_batch))
        return spec_batch, spec_pos_batch, spec_neg_batch, speaker_name_batch
