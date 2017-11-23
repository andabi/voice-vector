# -*- coding: utf-8 -*-
#!/usr/bin/env python


from audio import read, wav_random_crop, write
import os
import random
import glob
import tensorflow as tf


class DataLoader:

    def __init__(self, data_path, sr, duration, batch_size):
        self.data_path = data_path
        self.sr = sr
        self.duration = duration
        self.batch_size = batch_size
        self.speaker_names = [speaker for speaker in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, speaker))]

    def load_triplet(self, speaker_name):
        speakers_except = list(set(self.speaker_names) - set(list([speaker_name])))
        the_other_speaker_name = random.choice(speakers_except)
        wav_one_1 = self._get_randomly_cropped_wav_by_speaker(speaker_name)
        wav_one_2 = self._get_randomly_cropped_wav_by_speaker(speaker_name)
        wav_the_other = self._get_randomly_cropped_wav_by_speaker(the_other_speaker_name)
        return wav_one_1, wav_one_2, wav_the_other

    def _get_randomly_cropped_wav_by_speaker(self, speaker_name):
        wavfiles = glob.glob('{}/{}/*.wav'.format(self.data_path, speaker_name))
        wavfile = random.choice(wavfiles)
        randomly_cropped_wav = wav_random_crop(read(wavfile, self.sr), self.sr, self.duration)
        return randomly_cropped_wav

    def get_batch_queue(self):
        speaker_names = tf.convert_to_tensor(self.speaker_names)
        speaker_name = tf.train.slice_input_producer([speaker_names], shuffle=True)
        load_triplet = tf.py_func(self.load_triplet, speaker_name, (tf.float32, tf.float32, tf.float32))
        wav_one_1_batch, wav_one_2_batch, wav_the_other_batch = tf.train.batch(load_triplet,
                                                                   shapes=[(None, ), (None, ), (None, )],
                                                                   num_threads=32,
                                                                   batch_size=self.batch_size,
                                                                   capacity=self.batch_size * 32,
                                                                   dynamic_pad=True)
        return wav_one_1_batch, wav_one_2_batch, wav_the_other_batch

    def get_batch(self):
        pass

sr = 16000
batch_size = 2
duration = 1
data_loader = DataLoader('/Users/avin/git/speech_embedding/vcc2016_training', sr=sr, duration=duration, batch_size=batch_size)

a = random.choice(data_loader.speaker_names)

a_1, a_2, b = data_loader.get_batch_queue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    wav_a_1, wav_a_2, wav_b = sess.run([a_1, a_2, b])

    for i in range(batch_size):
        write(wav_a_1[i], sr, '/Users/avin/git/speech_embedding/a_1_{}.wav'.format(i))
        write(wav_a_2[i], sr, '/Users/avin/git/speech_embedding/a_2_{}.wav'.format(i))
        write(wav_b[i], sr, '/Users/avin/git/speech_embedding/b_{}.wav'.format(i))

    coord.request_stop()
    coord.join(threads)