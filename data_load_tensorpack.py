# -*- coding: utf-8 -*-
# !/usr/bin/env python
import glob
import multiprocessing
import os
import random

from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.prefetch import PrefetchData

from feature_extract import wav2melspec_db
from hparam import hparam as hp
from prepro import read_wav, crop_random_wav


class DataLoader(RNGDataFlow):
    def __init__(self, audio_meta, batch_size):
        self.audio_meta = audio_meta
        self.batch_size = batch_size
        self.speaker_dict = audio_meta.get_speaker_dict()

    def size(self):
        size = len(self.speaker_dict.keys())  # one batch is one step
        return size

    def get_data(self):
        while True:
            speaker_id = random.choice(self.speaker_dict.keys())
            wav = self._load_random_wav(speaker_id)
            mel_spec = wav2melspec_db(wav, hp.signal.sr, hp.signal.n_fft, hp.signal.win_length,
                                                             hp.signal.hop_length, hp.signal.n_mels)
            yield mel_spec, speaker_id

    def dataflow(self, nr_prefetch=1000, nr_thread=int(multiprocessing.cpu_count() // 1.5)):
        ds = self
        # TODO zero-padding
        ds = BatchData(ds, self.batch_size)
        ds = PrefetchData(ds, nr_prefetch, nr_thread)
        return ds

    def _load_random_wav(self, speaker_id):
        wavfile = self.audio_meta.get_random_audio(speaker_id)
        wav = read_wav(wavfile, hp.signal.sr)
        length = int(hp.signal.duration * hp.signal.sr)
        wav = crop_random_wav(wav, length=length)
        return wav  # (t, n_mel)


class AudioMeta:
    def __init__(self, data_path):
        self.data_path = data_path
        self.speaker_dict = dict(enumerate([speaker for speaker in os.listdir(data_path) if
                                            os.path.isdir(os.path.join(data_path,
                                                                       speaker))]))  # (k, v) = (speaker_id, speaker_name)
        self.audio_dict = dict()  # (k, v) = (speaker_id, wavfiles)

    def get_speaker_dict(self):
        return self.speaker_dict

    def num_speakers(self):
        return len(self.speaker_dict)

    def get_all_audio(self, speaker_id):
        if speaker_id not in self.audio_dict:
            wavfiles = glob.glob('{}/{}/*.wav'.format(self.data_path, self.speaker_dict[speaker_id]))
            self.audio_dict[speaker_id] = wavfiles
        return self.audio_dict[speaker_id]

    def get_random_audio(self, speaker_id):
        wavfiles = self.get_all_audio(speaker_id)
        wavfile = random.choice(wavfiles)
        return wavfile