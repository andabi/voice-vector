# -*- coding: utf-8 -*-
#!/usr/bin/env python


import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc

from hparam import hparam as hp
from modules import conv1d_banks, conv1d, normalize, highwaynet, gru
from tensorpack.train.tower import get_current_tower_context
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope


class Model(ModelDesc):
    '''
    n = batch size
    t = timestep size
    h = hidden size
    e = embedding size
    '''

    def __init__(self, num_banks, hidden_units, num_highway, norm_type, embedding_size):
        self.num_banks = num_banks
        self.hidden_units = hidden_units
        self.num_highway = num_highway
        self.norm_type = norm_type
        self.embedding_size = embedding_size

    def __call__(self):
        return self.y, self.speaker_id

    @auto_reuse_variable_scope
    def embedding(self, x, is_training=False):
        """
        :param x: shape=(n, t, n_mels)
        :return: embedding. shape=(n, e)
        """
        # Frame-level embedding
        x = tf.layers.dense(x, units=self.hidden_units, activation=tf.nn.relu)  # (n, t, h)

        out = conv1d_banks(x, K=self.num_banks, num_units=self.hidden_units, norm_type=self.norm_type,
                           is_training=is_training)  # (n, t, k * h)

        out = tf.layers.max_pooling1d(out, 2, 1, padding="same")  # (n, t, k * h)

        out = conv1d(out, self.hidden_units, 3, scope="conv1d_1")  # (n, t, h)
        out = normalize(out, type=self.norm_type, is_training=is_training, activation_fn=tf.nn.relu)
        out = conv1d(out, self.hidden_units, 3, scope="conv1d_2")  # (n, t, h)
        out += x  # (n, t, h) # residual connections

        for i in range(self.num_highway):
            out = highwaynet(out, num_units=self.hidden_units, scope='highwaynet_{}'.format(i))  # (n, t, h)

        out = gru(out, self.hidden_units, False)  # (n, t, h)

        # Take the last output
        out = out[..., -1]  # (n, h)

        # Final project for classification
        out = tf.layers.dense(out, self.embedding_size, name="final_projection")  # (n, c)
        out = tf.identity(out, name="embedding")

        return out

    def loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.speaker_id)
        loss = tf.reduce_mean(loss)
        return loss

    def accuracy(self):
        pred = tf.to_int32(tf.argmax(self.y, axis=1))
        acc = tf.reduce_mean(tf.to_float(tf.equal(self.speaker_id, pred)), name='accuracy')
        return acc

    def _get_inputs(self):
        length = hp.signal.duration * hp.signal.sr
        length_spec = length // hp.signal.hop_length + 1
        return [InputDesc(tf.float32, (None, length), 'wav'),
                InputDesc(tf.float32, (None, length_spec, hp.signal.n_mels), 'x'),
                InputDesc(tf.int32, (None,), 'speaker_id')]

    def _build_graph(self, inputs):
        _, self.x, self.speaker_id = inputs
        is_training = get_current_tower_context().is_training
        with tf.variable_scope('embedding'):
            self.y = self.embedding(self.x, is_training)  # (n, e)
        self.pred = tf.to_int32(tf.argmax(self.y, axis=1), name='prediction')
        self.cost = self.loss()

        # summaries
        tf.summary.scalar('train/loss', self.cost)
        tf.summary.scalar('train/accuracy', self.accuracy())

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=hp.train.lr, trainable=False)
        return tf.train.AdamOptimizer(lr)