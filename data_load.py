# -*- coding: utf-8 -*-
# !/usr/bin/env python


import glob
import os
import random

import tensorflow as tf
from hparam import Hparam

from audio import read_wav, wav_random_crop, wav2spectrogram, linear_to_mel, amp_to_db
import numpy as np


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
        self.speaker_dict = dict(enumerate([speaker for speaker in os.listdir(self.data_path) if
                                            os.path.isdir(os.path.join(self.data_path, speaker))]))

    def load_triplet(self, speaker_id):
        speakers_except = list(set(self.speaker_dict.keys()) - set(list([speaker_id])))
        the_other_speaker = random.choice(speakers_except)
        wav = self._get_random_wav(speaker_id)
        wav_pos = self._get_random_wav(speaker_id)
        wav_neg = self._get_random_wav(the_other_speaker)

        # Wav to spectrogram
        spec, _ = wav2spectrogram(wav, self.n_fft, self.win_length, self.hop_length)  # (1 + n_fft/2, t)
        spec_pos, _ = wav2spectrogram(wav_pos, self.n_fft, self.win_length, self.hop_length)
        spec_neg, _ = wav2spectrogram(wav_neg, self.n_fft, self.win_length, self.hop_length)

        # Mel-spectrogram
        spec = linear_to_mel(spec, self.sr, self.n_fft, self.n_mels)  # (n_mel, t)
        spec_pos = linear_to_mel(spec_pos, self.sr, self.n_fft, self.n_mels)
        spec_neg = linear_to_mel(spec_neg, self.sr, self.n_fft, self.n_mels)

        # # Decibel
        # spec = amp_to_db(spec)
        # spec_pos = amp_to_db(spec_pos)
        # spec_neg = amp_to_db(spec_neg)

        return spec.T, spec_pos.T, spec_neg.T  # (t, n_mel)

    def _get_random_wav(self, speaker_id):
        wavfiles = glob.glob('{}/{}/*.wav'.format(self.data_path, self.speaker_dict[speaker_id]))
        wavfile = random.choice(wavfiles)
        randomly_cropped_wav = wav_random_crop(read_wav(wavfile, self.sr), length=int(self.sr * self.duration))
        return randomly_cropped_wav

    def get_batch_queue(self):
        hp = Hparam.get_global_hparam()

        speaker_ids = tf.convert_to_tensor(self.speaker_dict.keys())
        speaker_id = tf.train.slice_input_producer([speaker_ids], shuffle=True)[0]
        spec, spec_pos, spec_neg = tf.py_func(self.load_triplet, [speaker_id], (tf.float32, tf.float32, tf.float32))
        spec_batch, spec_pos_batch, spec_neg_batch, speaker_id_batch = tf.train.batch(
            tensors=[spec, spec_pos, spec_neg, speaker_id],
            shapes=[
                (self.length_spec, self.n_mels),
                (self.length_spec, self.n_mels),
                (self.length_spec, self.n_mels), ()],
            num_threads=hp.data_load.num_threads,
            batch_size=self.batch_size,
            capacity=self.batch_size * hp.data_load.num_threads,
            dynamic_pad=True, name='batch_queue')
        return spec_batch, spec_pos_batch, spec_neg_batch, speaker_id_batch  # (n, t, n_mels)

    def get_batch_placeholder(self):
        spec_batch = tf.placeholder(tf.float32, shape=(self.batch_size, self.length_spec, self.n_mels), name='x')
        spec_pos_batch = tf.placeholder(tf.float32, shape=(self.batch_size, self.length_spec, self.n_mels),
                                        name='x_pos')
        spec_neg_batch = tf.placeholder(tf.float32, shape=(self.batch_size, self.length_spec, self.n_mels),
                                        name='x_neg')
        speaker_id_batch = tf.placeholder(tf.int32, shape=(self.batch_size,), name='speaker_id')
        return spec_batch, spec_pos_batch, spec_neg_batch, speaker_id_batch

    def get_batch(self):
        triplet_batch, speaker_id_batch = list(), list()
        for i in range(self.batch_size):
            speaker_id = random.choice(self.speaker_dict.keys())
            triplet = self.load_triplet(speaker_id)
            triplet_batch.append(triplet)
            speaker_id_batch.append(speaker_id)
        spec_batch, spec_pos_batch, spec_neg_batch = map(np.array, zip(*triplet_batch))
        return spec_batch, spec_pos_batch, spec_neg_batch, speaker_id_batch
