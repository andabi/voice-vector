# -*- coding: utf-8 -*-
# !/usr/bin/env python
import csv
import fnmatch
import glob
import os
import random
from datetime import datetime

from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.prefetch import PrefetchData

from audio import read_wav, crop_random_wav, fix_length
from audio import wav2melspec_db, normalize_db
from hparam import hparam as hp
from utils import split_path


class DataLoader(RNGDataFlow):
    def __init__(self, audio_meta, batch_size):
        self.audio_meta = audio_meta
        self.batch_size = batch_size
        self.speaker_dict = audio_meta.get_speaker_dict()

    def get_data(self):
        while True:
            speaker_id = random.choice(list(self.speaker_dict.keys()))
            wav = self._load_random_wav(speaker_id)
            mel_spec = wav2melspec_db(wav, hp.signal.sr, hp.signal.n_fft, hp.signal.win_length,
                                      hp.signal.hop_length, hp.signal.n_mels)
            mel_spec = normalize_db(mel_spec, max_db=hp.signal.max_db, min_db=hp.signal.min_db)
            yield wav, mel_spec, speaker_id

    def dataflow(self, nr_prefetch=1000, nr_thread=1):
        ds = self
        if self.batch_size > 1:
            ds = BatchData(ds, self.batch_size)
        ds = PrefetchData(ds, nr_prefetch, nr_thread)
        return ds

    def _load_random_wav(self, speaker_id):
        wavfile = self.audio_meta.get_random_audio(speaker_id)
        wav = read_wav(wavfile, hp.signal.sr)
        # wav = trim_wav(wav)
        length = int(hp.signal.duration * hp.signal.sr)
        wav = crop_random_wav(wav, length=length)
        wav = fix_length(wav, length)
        return wav  # (t, n_mel)


class AudioMeta(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.speaker_dict = self._build_speaker_dict(data_path)
        self.num_speaker = len(self.speaker_dict)
        self.audio_dict = dict()  # (k, v) = (speaker_id, wavfiles)

    def _build_speaker_dict(self, data_path):
        speaker_dict = dict(enumerate([s for s in sorted(os.listdir(data_path)) if os.path.isdir(
            os.path.join(data_path, s))]))  # (k, v) = (speaker_id, speaker_name)
        return speaker_dict

    def get_speaker_dict(self):
        return self.speaker_dict

    def num_speakers(self):
        return len(self.speaker_dict)

    def get_all_audio(self, speaker_id):
        if speaker_id not in self.audio_dict:
            path = '{}/{}'.format(self.data_path, self.speaker_dict[speaker_id])
            wavfiles = [os.path.join(dirpath, f) for dirpath, _, files in os.walk(path) for f in
                        fnmatch.filter(files, '*.wav')]
            # wavfiles = glob.glob('{}/{}/**/*.wav'.format(self.data_path, self.speaker_dict[speaker_id]))
            self.audio_dict[speaker_id] = wavfiles
        return self.audio_dict[speaker_id]

    def get_random_audio(self, speaker_id):
        wavfiles = self.get_all_audio(speaker_id)
        wavfile = random.choice(wavfiles)
        return wavfile


class VoxCelebMeta(AudioMeta):
    def __init__(self, data_path, meta_path=None):
        super(VoxCelebMeta, self).__init__(data_path=data_path)
        self.meta_dict = self._build_meta_dict(meta_path)

    def _build_meta_dict(self, meta_path):
        # field: full_name, sex, age, nationality, Job, Height, picture
        meta_dict = {}
        if not meta_path:
            return meta_dict

        with open(meta_path, 'rU') as f:
            reader = csv.DictReader(f)
            year = datetime.now().year
            for i, line in enumerate(reader):
                line['age'] = str(year - int(line['age']))
                meta_dict[i] = line
        return meta_dict

    def target_meta_field(self):
        return 'age', 'sex', 'nationality'


class CommonVoiceMeta(AudioMeta):
    def __init__(self, data_path, meta_path=None):
        super(CommonVoiceMeta, self).__init__(data_path=data_path)
        self.speaker_dict = self._build_speaker_dict(data_path)
        self.meta_dict = self._build_meta_dict(meta_path)

    def _build_speaker_dict(self, data_path):
        speaker_dict = dict(enumerate([split_path(s)[1] for s in sorted(glob.glob('{}/*.wav'.format(data_path)))]))
        return speaker_dict

    def _build_meta_dict(self, meta_path):
        # field: filename, text, up_votes, down_votes, age, gender, accent, duration
        meta_dict = {}
        if not meta_path:
            return meta_dict

        with open(meta_path, 'rb') as f:
            reader = csv.DictReader(f)
            for i, line in enumerate(reader):
                meta_dict[i] = line
        return meta_dict

    def get_all_audio(self, speaker_id):
        if speaker_id not in self.audio_dict:
            speaker = self.speaker_dict[speaker_id]
            wavfile = '{}/{}.wav'.format(self.data_path, speaker)
            self.audio_dict[speaker_id] = [wavfile]
        return self.audio_dict[speaker_id]

    def target_meta_field(self):
        return 'age', 'gender', 'accent'
