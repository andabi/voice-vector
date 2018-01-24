# -*- coding: utf-8 -*-
# !/usr/bin/env python

import os

import tensorflow as tf

from modules import conv1d_banks, conv1d, normalize, highwaynet, gru


class Model:
    '''
    n = batch size
    t = timestep size
    h = hidden size
    e = embedding size
    '''

    def __init__(self, data_loader, num_banks, hidden_units, num_highway, norm_type, embedding_size, is_training):
        self.data_loader = data_loader
        self.is_training = is_training
        self.num_banks = num_banks
        self.hidden_units = hidden_units
        self.num_highway = num_highway
        self.norm_type = norm_type
        self.embedding_size = embedding_size

        # Input
        self.x, self.speaker_id = data_loader.get_batch_queue()  # (n, t, n_mels)

        # Networks
        self.net = tf.make_template('net', self.embedding)
        self.y = self.net(self.x)  # (n, e)

    def __call__(self):
        return self.y, self.speaker_id

    def embedding(self, x):
        '''
        
        :param x: (n, t, n_mels)
        :return: (n, e)
        '''

        # Frame-level embedding
        x = tf.layers.dense(x, units=self.hidden_units, activation=tf.nn.relu)   # (n, t, h)

        out = conv1d_banks(x, K=self.num_banks, num_units=self.hidden_units, norm_type=self.norm_type,
                           is_training=self.is_training)  # (n, t, k * h)

        out = tf.layers.max_pooling1d(out, 2, 1, padding="same")  # (n, t, k * h)

        out = conv1d(out, self.hidden_units, 3, scope="conv1d_1")  # (n, t, h)
        out = normalize(out, type=self.norm_type, is_training=self.is_training, activation_fn=tf.nn.relu)
        out = conv1d(out, self.hidden_units, 3, scope="conv1d_2")  # (n, t, h)
        out += x  # (n, t, h) # residual connections

        for i in range(self.num_highway):
            out = highwaynet(out, num_units=self.hidden_units, scope='highwaynet_{}'.format(i))  # (n, t, h)

        out = gru(out, self.hidden_units, False)  # (n, t, h)

        # Take the last output
        out = out[..., -1]  # (n, h)

        # Final project for classification
        out = tf.layers.dense(out, len(self.data_loader.speaker_dict), name="final_projection")  # (n, c)

        return out

    def loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.speaker_id)
        loss = tf.reduce_mean(loss)
        return loss

    @staticmethod
    def load(sess, logdir):
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            tf.train.Saver().restore(sess, ckpt)
            model_name = Model.get_model_name(logdir)
            if model_name:
                print('Model loaded: {}'.format(model_name))
            else:
                print('Model created.')

    @staticmethod
    def get_model_name(logdir):
        path = '{}/checkpoint'.format(logdir)
        if os.path.exists(path):
            ckpt_path = open(path, 'r').read().split('"')[1]
            _, model_name = os.path.split(ckpt_path)
        else:
            model_name = None
        return model_name

    @staticmethod
    def all_model_names(logdir):
        import glob, os
        path = '{}/*.meta'.format(logdir)
        model_names = map(lambda f: os.path.basename(f).replace('.meta', ''), glob.glob(path))
        return model_names